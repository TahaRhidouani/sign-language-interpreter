# Sign Language Interpreter

### Main features:

- Can detect all 26 letters of the alphabet and the "space" character with over 98% accuracy
- Includes a clean & fast web interface to display translations in real-time
- Includes a python interface that can also be used as a module

<br />

**Recognizes characters in real-time** ![real-time](./images/screenshot_1.jpg)

**Features a tutorial modal to help with learning American Sign Language** ![tutorial](./images/screenshot_2.jpg)

## Run

The site is static (`views/`) and self-hosts its libraries: `package.json` pins MediaPipe Tasks Vision, TensorFlow.js and TypeIt, and `npm run vendor` copies their browser builds into `views/vendor/`. Every push to `main` builds the image `ghcr.io/taharhidouani/sign-language-interpreter` (tags `main` and `sha-<commit>`), which does that in a build stage and serves the result with Caddy on port 8080, answering `GET /healthz`. It needs no variables and stores nothing.

```sh
docker run -p 8080:8080 ghcr.io/taharhidouani/sign-language-interpreter:main
```

Locally, run `npm ci` and `npm run vendor`, then use any static file server, for example `python3 -m http.server -d views 8080`. The training code (`train/`, `dataset.csv`) is not part of the image.
