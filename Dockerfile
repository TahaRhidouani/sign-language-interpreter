FROM node:26.13.0-slim AS vendor
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund && npm run vendor

FROM caddy:2.11.9-alpine
ENV XDG_CONFIG_HOME=/tmp/caddy XDG_DATA_HOME=/tmp/caddy
COPY views/ /srv/
COPY --from=vendor /app/views/vendor/ /srv/vendor/
RUN touch /srv/healthz
USER nobody
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --start-interval=2s --retries=3 \
  CMD ["wget", "-q", "-O", "/dev/null", "http://127.0.0.1:8080/healthz"]
CMD ["caddy", "file-server", "--root", "/srv", "--listen", ":8080"]
