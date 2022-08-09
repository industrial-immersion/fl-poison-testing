## Docker image

To build:
  1. `docker run --privileged --rm tonistiigi/binfmt --install all`
     # https://github.com/docker/buildx/issues/495#issuecomment-991603416
  1. `docker dockerx create --use`
  1. `docker buildx build --platform linux/amd54,linux/arm64 -t
     nkxakouros/flwr-run:latest --push --pull .`
