# Sign Language Interpreter

### Main features:

- Can detect all 26 letters of the alphabet and the "space" character with over 98% accuracy
- Includes a clean & fast web interface to display translations in real-time
- Includes a python interface that can also be used as a module

<br />

**Recognizes characters in real-time** ![real-time](./images/screenshot_1.jpg)

**Features a tutorial modal to help with learning American Sign Language** ![tutorial](./images/screenshot_2.jpg)

## Run

The site is static (`views/`). Every push to `main` builds the image `ghcr.io/taharhidouani/sign-language-interpreter` (tags `main` and `sha-<commit>`), which serves `views/` with Caddy on port 8080 and answers `GET /healthz`. It needs no variables and stores nothing.

```sh
docker run -p 8080:8080 ghcr.io/taharhidouani/sign-language-interpreter:main
```

Locally, any static file server works, for example `python3 -m http.server -d views 8080`. The training code (`train/`, `dataset.csv`) is not part of the image.
