import pickle
from django.shortcuts import render

model = pickle.load(open("sentiment/model.pkl", "rb"))
vectorizer = pickle.load(open("sentiment/vectorizer.pkl", "rb"))

def home(request):

    prediction = None

    if request.method == "POST":

        review = request.POST.get("review")

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

    return render(request, "index.html", {"prediction": prediction})
