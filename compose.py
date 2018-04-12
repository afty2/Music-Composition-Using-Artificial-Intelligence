import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation

def compose():
    with open('dataset/pitchSet', 'rb') as filepath:
        pitches = pickle.load(filepath)

    pitchNames = sorted(set(item for item in pitches))
    pitchesLength = len(set(pitches))
    _input, inputNormalized = generatedSequenceFrom(pitches, pitchNames, pitchesLength)
    model = generatedNetworkFrom(inputNormalized, pitchesLength)
    output = outputGeneratedFrom(model, _input, pitchNames, pitchesLength)
    generateMidiFrom(output)

def generatedSequenceFrom(pitches, pitchNames, pitchesLength):
    pitchesToInts = dict((pitch, number) for number, pitch in enumerate(pitchNames))
    length = 100
    _input = []
    output = []
    for i in range(0, len(pitches) - length, 1):
        inSequence = pitches[i:i + length]
        outSequence = pitches[i + length]
        _input.append([pitchesToInts[char] for char in inSequence])
        output.append(pitchesToInts[outSequence])

    numberOfPatterns = len(_input)
    inputNormalized = numpy.reshape(_input, (numberOfPatterns, length, 1))
    inputNormalized = inputNormalized / float(pitchesLength)
    
    return (_input, inputNormalized)

def generatedNetworkFrom(_input, length):
    model = Sequential()
    model.add(LSTM(512,input_shape=(_input.shape[1], _input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(length))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.load_weights('weights/weights.hdf5')
                   
    return model

def outputGeneratedFrom(model, _input, pitchNames, length):
    start = numpy.random.randint(0, len(_input)-1)
    intsToPitches = dict((number, pitch) for number, pitch in enumerate(pitchNames))
    pattern = _input[start]
    output = []
    
    #NOTE:- range == 500 FOR NOW
    for pitchIndex in range(500):
        compositionInput = numpy.reshape(pattern, (1, len(pattern), 1))
        compositionInput = compositionInput / float(length)
        prediction = model.predict(compositionInput, verbose=0)
        index = numpy.argmax(prediction)
        result = intsToPitches[index]
        output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    
    return output

def generateMidiFrom(output):
    offSet = 0
    outputPitches = []
    
    for pattern in output:
        #NOTE:- Chords
        if ('.' in pattern) or pattern.isdigit():
            chordTones = pattern.split('.')
            pitches = []
            for tone in chordTones:
                pitch = note.Note(int(tone))
                pitch.storedInstrument = instrument.Piano()
                pitches.append(pitch)
            resultingChord = chord.Chord(pitches)
            resultingChord.offset = offSet
            outputPitches.append(resultingChord)
        #NOTE:- Pitches
        else:
            pitch = note.Note(pattern)
            pitch.offset = offSet
            pitch.storedInstrument = instrument.Piano()
            outputPitches.append(pitch)
        
        #NOTE:- offSet will change
        offSet += 0.5
    
    midiStream = stream.Stream(outputPitches)
    midiStream.write('midi', fp='output/composition.mid')

if __name__ == '__main__':
    compose()
