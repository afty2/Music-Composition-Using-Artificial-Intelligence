import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def beginTraining():
    pitchSet = parsePitches()
    pitchSetLength = len(set(pitchSet))
    input, output = generatedSequenceFrom(pitchSet, pitchSetLength)
    model = generateNetworkFrom(input, pitchSetLength)
    trainNetworkWith(model, input, output)

def parsePitches():
    pitches = []
    for file in glob.glob("MIDI/*.mid"):
        print("Parsing {0}".format(file))
        midi = converter.parse(file)
        parsablePitches = midi.flat.notes
        for element in parsablePitches:
            if isinstance(element, note.Note):
                pitches.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                pitches.append('.'.join(str(n) for n in element.normalOrder))

    with open('dataset/pitchSet', 'wb') as filepath:
        pickle.dump(pitches, filepath)
    
    return pitches

def generatedSequenceFrom(pitchSet, pitchSetLength):
    length = 100   #NOTE:-
    pitchNames = sorted(set(pitch for pitch in pitchSet))
    pitchesToInts = dict((pitch, number) for number, pitch in enumerate(pitchNames))
    input = []
    output = []
    for i in range(0, len(pitchSet) - length, 1):
        inSequence = pitchSet[i:i + length]
        outSequence = pitchSet[i + length]
        input.append([pitchesToInts[char] for char in inSequence])
        output.append(pitchesToInts[outSequence])
    
    numberOfPatterns = len(input)
    input = numpy.reshape(input, (numberOfPatterns, length, 1))
    input = input / float(pitchSetLength)
    output = np_utils.to_categorical(output)
    
    return (input, output)

def generateNetworkFrom(input, length):
    model = Sequential()
    model.add(LSTM(512, input_shape=(input.shape[1], input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(length))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
                   
    return model

def trainNetworkWith(model, input, output):
    filepath = "weights/weights.hdf5"
    checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=0,save_best_only=True,mode='min')
    callbacksData = [checkpoint]
    model.fit(input, output, epochs=200, batch_size=64, callbacks=callbacksData)

if __name__ == '__main__':
    beginTraining()
