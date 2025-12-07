# Jupyter stack with Python 3.7
# (base-notebook gives you start-notebook.sh, jovyan user, etc.)
FROM jupyter/base-notebook:python-3.7.6

USER root

# --- Java for PySpark ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-11-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME (path is standard on Debian/Ubuntu images)
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install PySpark 3.3.0 + your extra libs
RUN pip install --no-cache-dir \
    pyspark==3.3.0 \
    kafka-python \
    pymongo \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    plotly \
    folium \
    hdfs

# Back to jovyan for running the notebook server
USER ${NB_UID}
