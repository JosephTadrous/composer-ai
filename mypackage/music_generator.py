from mypackage import pitch_tokenizer
import numpy as np
import pandas as pd


class MusicGenerator:
    def __init__(self, model, scaler):
        self.model = model
        self.win = self.model.input_shape[0][1]
        self.scaler = scaler

    def generate(self, num):
        df = self.__create_df(size=num+self.win)
        self.__make_predictions(df, num)
        self.__transform(df)
        return df.iloc[self.win:]

    @staticmethod
    def __create_df(size):
        df = pd.DataFrame(0, index=np.arange(size), columns=['pitches', 'duration', 'velocity'])
        df.pitches = [np.zeros(108-21, dtype=int) for _ in range(size)]
        return df

    def __make_predictions(self, df, num):
        for i in range(num):
            df.iloc[i+self.win] = self.__predict(df.iloc[i:i+self.win])

    def __transform(self, df):
        df.pitches = pitch_tokenizer.detokenize_all(df.pitches)
        df[['duration', 'velocity']] = self.scaler.inverse_transform(df[['duration', 'velocity']])
        df.velocity = np.round(df.velocity).astype(int)

    def __predict(self, df):
        x_pitches = np.expand_dims(np.array([arr for arr in df['pitches']]), axis=0)
        x_dur_vel = np.expand_dims(df[['duration', 'velocity']].values, axis=0)
        pred_pitches, pred_dur_vel = self.model.predict([x_pitches, x_dur_vel])
        return self.__round_pitches(pred_pitches[0]), pred_dur_vel[:, 0], pred_dur_vel[:, 1]

    def __round_pitches(self, preds, temp=1.0):
        pitches = np.argwhere(preds >= 0.5).reshape(-1)
        if len(pitches) == 0:
            pitches = self.__sample_preds(preds.reshape(-1), temp)
        elif len(pitches) > 10:
            pitches = np.argsort(preds)[-10:]
        return pitch_tokenizer.tokenize(pitches + 21)

    @staticmethod
    def __sample_preds(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.atleast_2d(np.argmax(probas))