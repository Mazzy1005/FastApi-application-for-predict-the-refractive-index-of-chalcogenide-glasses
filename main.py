from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
from pytorch_tabular import TabularModel
from io import StringIO
from sklearn.decomposition import PCA

def augment_data(data):
    data["TeAs"] = data["Te"] * data["As"]
    data["GePb"] = data["Ge"] * data["Pb"]
    data["TeGe"] = data["Te"] * data["Ge"]
    data["TeTe"] = data["Te"] * data["Te"]
    data["AsAs"] = data["As"] * data["As"]
    data["AsGe"] = data["As"] * data["Ge"]
    data["GeGe"] = data["Ge"] * data["Ge"]
    data["SeSe"] = data["Se"] * data["Se"]
    data["SS"] = data["S"] * data["S"]
    return data


def data_compression(data, n=26):
    pca = PCA(n_components=n)
    X_pca = pd.DataFrame(pca.fit_transform(data))
    return X_pca


def get_composition_from_manual(data: str):
    elements = {'Li': [0.0], 'B': [0.0], 'O': [0.0], 'F': [0.0], 'Na': [0.0], 'Al': [0.0], 'Si': [0.0], 'P': [0.0], 'S': [0.0], 'Cl': [0.0], 'K': [0.0], 'Ca': [0.0], 'Mn': [0.0], 'Cu': [0.0], 'Zn': [0.0], 'Ga': [0.0], 'Ge': [0.0], 'As': [0.0], 'Se': [0.0], 'Br': [0.0], 'Ag': [0.0], 'Cd': [0.0], 'In': [0.0], 'Sn': [0.0], 'Sb': [0.0], 'Te': [0.0], 'I': [0.0], 'Cs': [0.0], 'Ba': [0.0], 'La': [0.0], 'Pr': [0.0], 'Gd': [0.0], 'Dy': [0.0], 'Er': [0.0], 'Tm': [0.0], 'Yb': [0.0], 'Hg': [0.0], 'Tl': [0.0], 'Pb': [0.0], 'Bi': [0.0]}
    data = data.replace(' ', '').split(",")
    for comp in data:
        comp = comp.split("-")
        elements[comp[0]][0] = float(comp[1])
    return augment_data(pd.DataFrame(elements))
    
    
def get_composition_from_csv(data: StringIO):
    elements = {'Li': [0.0], 'B': [0.0], 'O': [0.0], 'F': [0.0], 'Na': [0.0], 'Al': [0.0], 'Si': [0.0], 'P': [0.0], 'S': [0.0], 'Cl': [0.0], 'K': [0.0], 'Ca': [0.0], 'Mn': [0.0], 'Cu': [0.0], 'Zn': [0.0], 'Ga': [0.0], 'Ge': [0.0], 'As': [0.0], 'Se': [0.0], 'Br': [0.0], 'Ag': [0.0], 'Cd': [0.0], 'In': [0.0], 'Sn': [0.0], 'Sb': [0.0], 'Te': [0.0], 'I': [0.0], 'Cs': [0.0], 'Ba': [0.0], 'La': [0.0], 'Pr': [0.0], 'Gd': [0.0], 'Dy': [0.0], 'Er': [0.0], 'Tm': [0.0], 'Yb': [0.0], 'Hg': [0.0], 'Tl': [0.0], 'Pb': [0.0], 'Bi': [0.0]}
    data = pd.read_csv(data, sep=";")
    print(data)
    for i in data.columns:
        if i in elements:
            elements[i] = data[i].to_list()
    return augment_data(pd.DataFrame.from_dict(elements, orient="index").T.fillna(0))


app = FastAPI()

model = TabularModel.load_model("simple_model")


@app.get("/")
def return_start_page():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict(data = Body(...)):
    res = {}
    print(data)
    if data["type"] == "manual":
        composition = get_composition_from_manual(data["composition"])
        res["prediction"] = model.predict(composition)["ND300_prediction"].to_list()[0]
    elif data["type"] == "csv":
        composition = get_composition_from_csv(StringIO(data["data"].replace(',', '.')))
        res["predictions"] = model.predict(composition)["ND300_prediction"].to_list()
    return res

@app.get("/hello")
def hello():
    return "Hello"