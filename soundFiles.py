def soundfile(soundFile):
    sound_files = {
        "clipSpeech" : "/soundFiles/clips/newsweekonair10sec.flac",
        "clipRock" : "/soundFiles/clips/rock10sec.flac",
        "clipPop" : "/soundFiles/clips/popstars10sec.flac",
        "clipClassic" : "/soundFiles/clips/orchestra10sec.flac",
        "clipRock2" : "/soundFiles/clips/rock10sec2.flac",
        "clipSpeechMP3"  : "/soundFiles/codecs/newsweekonair10sec.mp3",
        "clipSpeech1secMP3"  : "/soundFiles/codecs/newsweekonair1sec.mp3",
        "clipSpeechOPUS"  : "/soundFiles/codecs/newsweekonair10sec.opus",
        "clipSpeech1secOPUS"  : "/soundFiles/codecs/newsweekonair1sec.opus",
        "clipClassicalMP3"  : "/soundFiles/codecs/orchestra10sec.mp3",
        "clipClassical1secMP3"  : "/soundFiles/codecs/orchestra1sec.mp3",
        "clipClassicalOPUS"  : "/soundFiles/codecs/orchestra10sec.opus",
        "clipClassical1secOPUS"  : "/soundFiles/codecs/orchestra1sec.opus",
        "clipRockMP3"  : "/soundFiles/codecs/rock10sec.mp3",
        "clipRock1secMP3"  : "/soundFiles/codecs/rock1sec.mp3",
        "clipRockOPUS"   : "/soundFiles/codecs/rock10sec.opus",
        "clipRock1secOPUS"  : "/soundFiles/codecs/rock1sec.opus",
        "clipPopMP3"  : "/soundFiles/codecs/popstars10sec.mp3",
        "clipPop1secMP3"  : "/soundFiles/codecs/popstars1sec.mp3",
        "clipPopOPUS"  : "/soundFiles/codecs/popstars10sec.opus",
        "clipPop1secOPUS"  : "/soundFiles/codecs/popstars1sec.opus"
    }

    if soundFile in sound_files:
        return sound_files[soundFile]
    else:
        print(f"Sound file {soundFile} not found!!")
        print("Available sound files: ")
        for key, value in sound_files.items():
            print(key, ": ", value)