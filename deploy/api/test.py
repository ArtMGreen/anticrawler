import requests


def test_case(url, request_type, files=None):
    if request_type == 'GET':
        resp = requests.get(url)
    elif request_type == 'POST':
        resp = requests.post(url, files=files)

    print(resp)
    print(resp.content)
    print(resp.headers)
    print(resp.history)

#
# test_case(
#     'http://localhost:8000/upload','POST',
#     {'file': open('2nf26.png', 'rb')}
# )

get_urls = [
    # "http://localhost:8000/attack/?method=PGD&filename=2nf26.png",
    # "http://localhost:8000/defend/?method=fdfffffffff&filename=2nf26.png",
    # "http://localhost:8000/inference/?filename=2nf26.png",
    "http://localhost:8000/defend/?method=MEDIAN_FILTER&filename=2nf26.png",
    "http://localhost:8000/defend/?method=THRESHOLDING&filename=2nf26.png",
    "http://localhost:8000/defend/?method=GRADIENT_TRANSFORM&filename=2nf26.png"
]

for url in get_urls:
    test_case(url, 'GET')