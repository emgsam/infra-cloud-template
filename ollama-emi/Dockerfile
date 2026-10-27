# Carga de IA de la Clase 1: Ollama + llama3.2:1b.
# Los pesos van AFUERA: se descargan en runtime a un volumen en /root/.ollama.
FROM ollama/ollama:0.40.11

ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODEL=llama3.2:1b

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 11434

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=60 \
  CMD ollama show "$OLLAMA_MODEL" > /dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
