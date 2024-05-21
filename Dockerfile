# Используем базовый образ Ubuntu
FROM ubuntu:20.04

# Устанавливаем необходимые зависимости
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk wget unzip xvfb x11vnc mesa-utils curl && \
    rm -rf /var/lib/apt/lists/*

# Создаем директорию /opt, если она не существует
RUN mkdir -p /opt

# Скачиваем и устанавливаем Apache Spark
RUN curl -o /tmp/spark.tgz https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop3.2.tgz && \
    tar -xzf /tmp/spark.tgz -C /opt && \
    rm /tmp/spark.tgz && \
    mv /opt/spark-3.0.0-bin-hadoop3.2 /opt/spark

# Скачиваем и устанавливаем JavaFX 17
RUN wget https://download2.gluonhq.com/openjfx/17.0.1/openjfx-17.0.1_linux-x64_bin-sdk.zip -O /tmp/javafx.zip && \
    unzip /tmp/javafx.zip -d /opt && \
    rm /tmp/javafx.zip

# Копируем JAR файл вашего приложения и конфигурацию Log4j в контейнер
COPY target/Test27JavaFX-1.0-SNAPSHOT.jar /app/Test27JavaFX-1.0-SNAPSHOT.jar
COPY src/main/resources/log4j.properties /app/log4j.properties

# Переходим в директорию с приложением
WORKDIR /app

# Устанавливаем главный класс для Spark
ENV SPARK_APPLICATION_MAIN_CLASS=AnomalyDetectorAppAdminGUI
ENV SPARK_MASTER_URL=spark://spark-master:7077
ENV PATH_TO_FX=/opt/javafx-sdk-17.0.1/lib
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV LIBGL_ALWAYS_SOFTWARE=true
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin

# Запускаем Xvfb и x11vnc, затем запускаем Spark приложение
CMD Xvfb :99 -screen 0 1024x768x24 & \
    x11vnc -display :99 -N -forever -nopw & \
    export DISPLAY=:99 && \
    $SPARK_HOME/bin/spark-submit --master ${SPARK_MASTER_URL} --class ${SPARK_APPLICATION_MAIN_CLASS} /app/Test27JavaFX-1.0-SNAPSHOT.jar --module-path ${PATH_TO_FX} --add-modules javafx.controls,javafx.fxml