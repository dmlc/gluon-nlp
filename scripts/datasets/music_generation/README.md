# Music Generation

We provide datasets for training a music generation model. 

## Maestro

See https://magenta.tensorflow.org/datasets/maestro for detailed introduction.

```
# Get V1 Dataset
nlp_data prepare_music_midi --dataset maestro_v1

# Get V2 Dataset
nlp_data prepare_music_midi --dataset maestro_v2
```

## LakhMIDI

See https://colinraffel.com/projects/lmd/ for more details

```
# Get Lakh MIDI Full Dataset
nlp_data prepare_music_midi --dataset lmd_full

# Get the subset of 45,129 files from LMD-full 
# which have been matched to entries in the Million Song Datase
nlp_data prepare_music_midi --dataset lmd_matched

# Get the aligned version of lmd_matched
nlp_data prepare_music_midi --dataset lmd_aligned

# Get the clean midi data
nlp_data prepare_music_midi --dataset clean_midi
```

## Geocities

The Geocities collection of MIDI files. 
See https://archive.org/details/archiveteam-geocities-midi-collection-2009 for more details.
```
nlp_data prepare_music_midi --dataset geocities
```
