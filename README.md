# Speech-emotion-recognition
End-to-End Speech Emotion Recognition Project using various datasets to classify speech audio into different categories of emotions.

## Datasets used: 
- "[The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/records/1188976)" by Livingstone & Russo is licensed under CC BY-NA-SC 4.0.
- "[CREMA-D]" by [David Cooper Cheyney](https://github.com/CheyneyComputerScience/CREMA-D)
- "[Surrey Audio-Visual Expressed Emotion (SAVEE)]" by [University of Surrey](http://kahlan.eps.surrey.ac.uk/savee/Database.html).
- "[Toronto emotional speech set (TESS)]" by [University of Toronto](https://tspace.library.utoronto.ca/handle/1807/24487).

## Emotion Categories:
| Code | Emotion |   | Code | Emotion |
|------|---------|:-:|------|---------|
| 1    | Neutral |   | 5    |  Fear   |
| 2    |  Happy  |   | 6    | Disgust |
| 3    |   Sad   |   | 7    |Surprise | 
| 4    |  Angry  |   |      |         |

## Models experimented with:
- CNN
- Transformer
- NN
- LSTM

## Accuracy Metrics:
| Model            | Epochs | Accuracy |
|------------------|--------|----------|
| CNN              | 50     | 97%      | 
| CLSTM            | 130    | 82%      | 
| Neural Network   | 1000   | 80%      |  
| Transformer      | 500    | 75%      |  


<br> Best model deployed locally using **Streamlit**.

## Contributing members:
- [Saikat Bera](https://github.com/berasaikat)
- [Shreyan Chakraborty](https://github.com/shreyanc07)
- [Sayantan Mondal](https://github.com/msayantanm)
- [Shubhangi Sanyal](https://github.com/ShubhangiSanyal)
