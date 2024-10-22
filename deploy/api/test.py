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


# test_case(
#     'http://localhost:8000/upload','POST',
#     {'file': open('some_absplute_path', 'rb')}
# )

get_urls = [
    "http://localhost:8000/attack/?method=PGD&filename=2pfpn.png",
    "http://localhost:8000/defend/?method=fdfffffffff&filename=2cg58.png",
    "http://localhost:8000/inference/?filename=2cg58.png"
]

for url in get_urls:
    test_case(url, 'GET')