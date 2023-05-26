
# BLOOM INFERENCE SERVER SIDE
## Install manualy
Install package dependence
```
pip install flask flask_api gunicorn pydantic accelerate huggingface_hub>=0.9.0 deepspeed>=0.7.3 deepspeed-mii==0.0.2
```

## Start server
```
make bloom-560m
```

## Test request

```
python server_request.py
```

Or you can start using docker

```
sudo docker compose up -d
```


## Progress
- [x] Basic API server side
- [x] Using HF and DS for serving
