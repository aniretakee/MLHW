from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import sklearn
import numpy as np
from typing import Dict


app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

    def to_dict(self)->Dict:
        return dict(name=self.name,
                    year=self.year,
                    selling_price=self.selling_price,
                    km_driven=self.km_driven,
                    fuel=self.fuel,
                    seller_type=self.seller_type,
                    transmission=self.transmission,
                    owner=self.owner,
                    mileage=self.mileage,
                    engine=self.engine,
                    max_power=self.max_power,
                    torque=self.torque,
                    seats=self.seats)




class Items(BaseModel):
    objects: List[Item]

    def __getitem__(self, item):
        return getattr(self, item)

db:List[Items] = [Item(
                        name= 'a',
                        year=0,
                        selling_price=0,
                        km_driven=0,
                        fuel='a',
                        seller_type='a',
                        transmission='a',
                        owner='a',
                        mileage='a',
                        engine='a',
                        max_power='a',
                        torque='a',
                        seats=0,)
                 ]

@app.get('/')
async def root():
    return {'message': 'Hello'}

def fapi_predict(
            json_ex,
            normalizer=pickle.load(open(r'normalizer.sav', 'rb')),
            model=pickle.load(open(r'numerical_model.sav', 'rb')),
            items_=False):

    df = pd.read_json(json_ex)

    '''Очистка кода'''
    df = df.drop(columns=['name', 'selling_price', 'torque'])

    df['mileage'] = df['mileage'].str.replace(r"[^\d\.]", "", regex=True)  # удаляем все символы (кроме цифр и точки)
    df['engine'] = df['engine'].str.replace(r"[^\d\.]", "", regex=True)
    df['max_power'] = df['max_power'].str.replace(r"[^\d\.]", "", regex=True)
    df['mileage'] = pd.to_numeric(df['mileage'])  # приводим к типу float64
    df['engine'] = pd.to_numeric(df['engine'])
    df['max_power'] = pd.to_numeric(df['max_power'])

    if items_:
        bad_cols = ['mileage', 'engine', 'max_power', 'seats']

        for col in bad_cols:
            df[col].fillna(df[col].median(), inplace=True)
            print(f'Пропусков в столбце {col} = {df[col].isna().sum()}')

    df['seats'] = df['seats'].astype('int')

    '''Разделение данных для нормализации'''
    cat_features_mask = (df.dtypes == "object").values  # категориальные признаки имеют тип "object"
    df_num = df[df.columns[~cat_features_mask]]
    x_num_norm = normalizer.fit_transform(df_num)
    df_num_norm = pd.DataFrame(data=x_num_norm)

    if not items_:
        return model.predict(df_num_norm)
    else:
        list_= model.predict(df_num_norm)
        return ' '.join([str(elem) for elem in list_])

@app.post("/predict_item")
def predict_item(dict_:Dict) -> float:
    df = pd.DataFrame.from_dict(dict_)
    json_ = df.to_json()
    return fapi_predict(json_)[0]


@app.post("/predict_items")
def predict_items(items: Dict):# -> List[float]:
    df = pd.DataFrame.from_dict(items)
    json_ = df.to_json()
    return fapi_predict(json_, items_=True)





