import argparse

import requests


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--host", type=str,default="localhost", help="host address")
    group.add_argument("--port", type=int,default=5000, help="port number")

    return parser.parse_args()


def generate(url: str) -> None:
    url = url + "/generate/"

    request_body = {
        "text": [
            "Deep speed",
        ],
        "max_new_tokens": 1000,
    }
    response = requests.post(url=url, json=request_body, verify=False)
    print(response.json(), "\n")



def main():
    args = get_args()
    url = "http://{}:{}".format(args.host, args.port)

    generate(url)


if __name__ == "__main__":
    main()
