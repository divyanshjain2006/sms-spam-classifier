mkdir -p ~/.streamlit/
printf "[server]\nport = %s\nenableCORS = false\nheadless = true\n" "${PORT:-8501}" > ~/.streamlit/config.toml
