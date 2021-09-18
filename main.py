#The header 44 bytes sets: ChunkID, ChunkSize, Format,SubChunk1ID, Subchunk1Size,Audioformat,Numchannels, Samplerate,ByteRate,BlockAlign,BitsPerSample,Subchunk2ID,Subchunk2Size,data
#Audio is split by channels > frames > samplewidth of frames

import wave, numpy, struct
from scipy.fftpack import fft

import sys
import time

def readAudioData(wave_name):

    waveFile=wave.open(wave_name,'rb')#Read binary file
    frameNum=waveFile.getnframes()
    sampleWidth=waveFile.getsampwidth()
    numChannels=waveFile.getnchannels()
    samplingRate=waveFile.getframerate()
    s=''
    if sampleWidth==3:
        for i in range(frameNum):
            wavedata=waveFile.readframes(1) #Take next frame
            for c in range(0,3*numChannels,3):
                s += '\0'+wavedata[c:(c+3)] # put TRAILING 0 to make 32-bit (file is little-endian)
                        #data=struct.unpack('hh',wavedata)#unpack binary file as short
    else:
            s = waveFile.readframes(frameNum)
    waveFile.close()
    unpstr= '{0}{1}'.format(numChannels*frameNum, {1:'b',2:'h',3:'i',4:'i',8:'q'}[sampleWidth])
    x = list(struct.unpack(unpstr, s))
    if sampleWidth == 3:
        x = [k >> 8 for k in x] #downshift to get +/- 2^24 with sign extension
    return [x,samplingRate,numChannels]
readAudioData('Test.wav')

def goertzel_mag(numSamples,target_freq,sampling_rate, wave_data):
    #Goertzel filter: Weirdly enough, it uses a similar method to the matched filter. The maximum magnitude energy of the signal is the detected frequency.

    scalingFactor= numSamples / 2.0 # 2 channels
    k= int((0.5+ ((numSamples * target_freq)/ sampling_rate)))
    omega = (2.0 * numpy.pi * k) / numSamples
    q0,q1,q2 = [0,0,0]


    for i in range(int(scalingFactor)):
        q0 = 2.0 * numpy.cos(omega) * q1 - q2 + wave_data[(2*i)] #wave_data is stereo
        q2 = q1
        q1 = q0
    real = (q1 - q2 * numpy.cos(omega)) / scalingFactor
    imag = (q2 * numpy.sin(omega)) / scalingFactor

    magnitude = numpy.sqrt(real**2 + imag**2)
    return magnitude
    #https://stackoverflow.com/questions/11579367/implementation-of-goertzel-algorithm-in-c

def readDMFTtones(wave_file):
    #This function reads the .wav file every .1 seconds and puts what it thinks the tone is into an array.
    #From there, if the tone repeats, it is assumed that it was the same tone previously and will be dealt with in ____ function

    [wave_data,samplingRate,numChannels]=readAudioData(wave_file)
    numSamples=int(samplingRate * .25 * numChannels)   #.25 seconds, stereo   #int(len(wave_data)/2)

    row_freq =[1209,1336,1477,1633]
    col_freq=[697,770,852,941]
    matrix=[['1','2','3','A'],['4','5','6','B'],['7','8','9','C'],['*','0','#','D']]

    tones=[]
    filtered_tones=[]

    for section in range(int(len(wave_data)/numSamples)):
        row_magnitudes= numpy.zeros(len(row_freq))
        col_magnitudes= numpy.zeros(len(col_freq))

        for freq in range(len(row_freq)):
            this=wave_data[section*numSamples:(section+1)*numSamples]
            row_magnitudes[freq]=goertzel_mag(numSamples,row_freq[freq],samplingRate,wave_data[section*numSamples:(section+1)*numSamples])
        for freq in range(len(col_freq)):
            col_magnitudes[freq]=goertzel_mag(numSamples,col_freq[freq],samplingRate,wave_data[section*numSamples:(section+1)*numSamples])

        if max(row_magnitudes) > .01 and  max(col_magnitudes) > .01:
            #Finds the max value of each magnitude and uses those values to look up the DTMF tone.
            #also numpy arrays are a pain to deal with.
            tones.append(matrix[int(numpy.where(col_magnitudes==numpy.amax(col_magnitudes))[0])][int(numpy.where(row_magnitudes==numpy.amax(row_magnitudes))[0])])
        else:
            print ('DMFTtone not found')
            tones.append(' ')

    n=1
    #If the current tone is the same as the previous tone or a tone in a previous part of the list, delete said tone.
    for array_tone in range(len(tones)):
        if array_tone == 0:
            filtered_tones.append(tones[array_tone])
            continue
        elif tones[array_tone-n] == tones[array_tone]:
            n += 1 #still wish python had a n++ operator.
            tones[array_tone]=''
            continue
        else:
            n=1 #reset the counter.
            filtered_tones.append(tones[array_tone])
    return(filtered_tones)
#print(readDMFTtones('Test.wav'))
def findMatLocation(matrix,string):
    ##To do, add a condition that parses input.
    #I saw some C code that does this via bitwise operations. It's probably a hell of a lot faster, but I already made this.

    matrix_dim=len(matrix[0])
    item_index=0
    for row in matrix:
        for i in row:
            if i== string:
                break
            item_index +=1
        if i==string:
            break
    return [(item_index%matrix_dim),(int(item_index/ matrix_dim))] #(x,y) or (row,column)
#remember that python starts index @ zero.

def createDTMFtone(string_num, wav_name,duration):
    #        |1209 Hz |  1336 Hz   |  1477Hz   |   1633 Hz  |
    # 697 Hz|   [1]       [2]          [3]          [A]
    # 770 Hz|   [4]       [5]          [6]          [B]
    # 852 Hz|   [7]       [8]          [9]          [C]
    # 941 Hz|   [*]       [0]          [#]          [D]

    row_freq =[1209,1336,1477,1633]
    col_freq=[697,770,852,941]
    matrix=[['1','2','3','A'],['4','5','6','B'],['7','8','9','C'],['*','0','#','D']]

    sampleRate=44100
    #duration = 2 # seconds
    nchannels = 2
    sampwidth=2


    #This could be range as well, but apparently linspace always includes endpoints while range does not.
    #Also, the numpy convolve function requires an array-like object.
    shared_time_axis=numpy.linspace(0,duration,duration*sampleRate) #(start,stop,num_samples)

    #full_signal=numpy.zeros(int(len(string_num)*sampleRate*duration)) #pre-allocate

    wave_obj=wave.open(wav_name, 'w')
    wave_obj.setparams((nchannels, sampwidth, sampleRate, 0, 'NONE', 'not compressed'))
    for n in range(len(string_num)):
        [freq_1_loc,freq_2_loc]=findMatLocation(matrix,string_num[n])
        [freq_1,freq_2]=[row_freq[freq_1_loc],col_freq[freq_2_loc]]

        #To properly zero pad the fft, you need to have each vector length to be [length vector_len1+vector_len2-1]
        #Similarly, it would be helpful to have a window for there to be less spectral leakage from the fft. I chose a Hann window arbitrarily, but this window is suggested for sine waves.

        signal_1=numpy.zeros(2*len(shared_time_axis)-1) #pre-allocate. Both vector_1 and vector_2 are same length (shared_time_axis)
        signal_2=numpy.zeros(2*len(shared_time_axis)-1)

        signal_1[0:int(duration*sampleRate)]=(numpy.sin(freq_1 * 2 * numpy.pi * shared_time_axis))#*numpy.hanning(len(shared_time_axis)))
        signal_2[0:int(duration*sampleRate)]=(numpy.sin(freq_2 * 2 * numpy.pi * shared_time_axis))#*numpy.hanning(len(shared_time_axis)))

        #Create two sine waves in time domain and convolve together,
        #or two impulses in the frequency domain, multiply, then perform ifft.
        #Granted, if you convolve you do get a longer signal (2n-1).
        #https://www.mathworks.com/matlabcentral/answers/38066-difference-between-conv-ifft-fft-when-doing-convolution

        #Just as expected, the fft was a lot faster using time library.
        fft_signal=numpy.fft.ifft(numpy.fft.fft(signal_1) * numpy.fft.fft(signal_2))


        #print(numpy.size(fft_signal))
        #Test=numpy.convolve(signal_1,signal_2)
        #print(sum(numpy.convolve(signal_1,signal_2) != 0)) #Mask to remove zero points in array from conv.



        max_amplitude = float(int((2 ** (sampwidth * 8)) / 2) - 1) #sampwidth=2, so max amplitude=32767
        max_fft_signal=max(fft_signal)
        for i in range(len(fft_signal)):
            left_right_channel = int(32767.0*fft_signal[i]/max_fft_signal)
            wave_obj.writeframesraw( struct.pack('<hh', left_right_channel, left_right_channel) )

    wave_obj.close()
createDTMFtone('AC','Test.wav',1)

##GUI inspiration taken from https://github.com/markjay4k/Audio-Spectrum-Analyzer-in-Python
