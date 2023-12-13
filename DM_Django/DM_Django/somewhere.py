def handle_uploaded_file(f):
    with open("static/dataset/input/input.csv", "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)