import json
import requests
data = {
    "temperature": 3068,
    "luminosity":0.0024,
    "radius": 0.17,
    "absolute_magnitude":16.12,
    "star_color": "Red",
    "spectral_class":"M"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.text)