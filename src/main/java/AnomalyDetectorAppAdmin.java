import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;

import static org.apache.spark.sql.functions.col;

public class AnomalyDetectorAppAdmin {
    public static void main(String[] args) {
        //Logger.getLogger("org.apache.spark").setLevel(Level.ERROR);

        // Создание SparkSession
        SparkSession spark = SparkSession.builder()
                .appName("AnomalyDetectorAppAdmin")
                .config("spark.master", "local")
                .getOrCreate();

        // Получение настроек от пользователя через GUI (для примера пока задано жестко)
        String dataSourceType = "file"; // Получение этого значения из GUI
        String fileType = "csv"; // Получение этого значения из GUI
        String outputType = "file"; // Можно будет заменить на ввод с GUI ("file" или "kafka")
        String outputDestination = "results.txt"; // или можно заменить на ввод с GUI (для файла)
        String kafkaBrokers = "localhost:9092"; // или можно заменить на ввод с GUI (для Kafka)
        String kafkaTopic = "anomalies"; // или можно заменить на ввод с GUI (для Kafka)
        String algorithm = "kMeans"; // Можно выбрать "kMeans" или "bisectingKMeans" через GUI
        String anomalyMetric = "Manhattan"; // Можно выбрать "Euclidean", "Manhattan" или другие метрики через GUI
        String[] selectedFeatures = new String[]{"Genre", "Age", "AnnualIncome", "SpendingScore"}; // Получение этих значений из GUI

        // Выбор источника данных
        DataSource dataSource;
        if (dataSourceType.equalsIgnoreCase("file")) {
            dataSource = FileDataSourceSelector.createFileDataSource(fileType, "C:\\Users\\Elena Shustova\\Desktop\\FOLDER\\SGTU\\DIPLOM\\Mall_Customers train.csv", spark);
        } else {
            dataSource = DataSourceSelector.createDataSource(dataSourceType, null, kafkaBrokers, kafkaTopic, spark);
        }

        if (dataSource == null) {
            throw new IllegalArgumentException("Неподдерживаемый тип источника данных или файла.");
        }

        Dataset<Row> data = dataSource.getData();
        System.out.println("Считанные данные:");
        data.show();

        // Препроцессинг данных
        //Dataset<Row> preprocessedData = preprocessData(data, selectedFeatures);

        // Выбор алгоритма кластеризации
        ClusterizerSelector clusterizerSelect = new ClusterizerSelector();
        Clusterizer<?> clusterizer = clusterizerSelect.createClusterizer(algorithm);

        // Применение выбранного алгоритма кластеризации к данным из файла
        Model<?> clusteringModel = clusterizer.cluster(data, selectedFeatures);

        // Создание экземпляра ResultWriter в зависимости от выбора пользователя
        ResultWriter resultWriter = ResultWriterSelector.createResultWriter(outputType, outputDestination, kafkaBrokers, kafkaTopic);

        // Выбор метрики для обнаружения аномалий
        DistanceMetric distanceMetric = DistanceMetricSelector.createMetric(anomalyMetric);

        // Создание экземпляра класса AnomalyDetector с выбранными признаками
        AnomalyDetector<?> anomalyDetector = new AnomalyDetector<>(clusteringModel, resultWriter, distanceMetric, selectedFeatures);

        // Новые данные для обнаружения аномалий
        DataSource newDataSource;
        if (dataSourceType.equalsIgnoreCase("file")) {
            newDataSource = FileDataSourceSelector.createFileDataSource(fileType, "C:\\Users\\Elena Shustova\\Desktop\\FOLDER\\SGTU\\DIPLOM\\Mall_Customers test.csv", spark);
        } else {
            newDataSource = DataSourceSelector.createDataSource(dataSourceType, null, kafkaBrokers, kafkaTopic, spark);
        }

        if (newDataSource == null) {
            throw new IllegalArgumentException("Неподдерживаемый тип источника данных или файла.");
        }

        Dataset<Row> newData = newDataSource.getData();
        System.out.println("Новые данные для обнаружения аномалий:");
        newData.show();

        // Препроцессинг новых данных
        //Dataset<Row> preprocessedNewData = preprocessData(newData, selectedFeatures);

        // Обработка и обнаружение аномалий на новых данных
        anomalyDetector.detectAnomalies(newData);

        // Закрытие SparkSession
        spark.stop();
    }

    public static Dataset<Row> preprocessData(Dataset<Row> data, String[] selectedFeatures) {
        // Выбор только выбранных признаков из исходного датасета
        Column[] selectedCols = Arrays.stream(selectedFeatures)
                .map(colName -> col(colName))
                .toArray(Column[]::new);

        Dataset<Row> selectedData = data.select(selectedCols);

        // Индексация строковых столбцов выбранных признаков
        String[] stringColumns = Arrays.stream(selectedFeatures)
                .filter(col -> selectedData.schema().apply(col).dataType().typeName().equalsIgnoreCase("string"))
                .toArray(String[]::new);

        Dataset<Row> indexedData = selectedData;

        // Индексация строковых столбцов
        for (String column : stringColumns) {
            StringIndexer stringIndexer = new StringIndexer()
                    .setInputCol(column)
                    .setOutputCol(column + "_index")
                    .setHandleInvalid("skip");

            indexedData = stringIndexer.fit(indexedData).transform(indexedData);
        }

        // Удаление строковых столбцов из итогового датасета
        for (String column : stringColumns) {
            indexedData = indexedData.drop(column);
        }
        System.out.println("Содержимое indexedData после индексации и удаления строковых столбцов:");
        indexedData.show();
        indexedData.printSchema();

        // Преобразование векторных данных
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(indexedData.columns())
                .setOutputCol("assembledFeatures");

        Dataset<Row> assembledData = assembler.transform(indexedData);

        // Масштабирование данных
        StandardScaler scaler = new StandardScaler()
                .setInputCol("assembledFeatures")
                .setOutputCol("scaledFeatures");

        Dataset<Row> scaledData = scaler.fit(assembledData).transform(assembledData);

        System.out.println("Содержимое scaledData после масштабирования:");
        scaledData.show();
        scaledData.printSchema();

        indexedData = scaledData;

        // Возвращение итогового датасета
        return indexedData;
    }
}

// Интерфейс для источника данных
interface DataSource {
    Dataset<Row> getData();
}

// Класс для работы с CSV файлом
class CsvFileDataSource implements DataSource {
    private final String filePath;
    private final SparkSession spark;

    CsvFileDataSource(String filePath, SparkSession spark) {
        this.filePath = filePath;
        this.spark = spark;
    }

    @Override
    public Dataset<Row> getData() {
        Dataset<Row> data = spark.read()
                .format("csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(filePath);

        System.out.println("Считанные данные из CSV файла:");
        data.show();
        return data;
    }
}

// Класс для выбора типа файла
class FileDataSourceSelector {
    public static DataSource createFileDataSource(String fileType, String filePath, SparkSession spark) {
        switch (fileType.toLowerCase()) {
            case "csv":
                return new CsvFileDataSource(filePath, spark);
            // Добавить другие типы файлов здесь
            default:
                throw new IllegalArgumentException("Неподдерживаемый тип файла: " + fileType);
        }
    }
}

// Класс для работы с Kafka источником данных
class KafkaDataSource implements DataSource {
    private final String kafkaBrokers;
    private final String topic;
    private final SparkSession spark;

    KafkaDataSource(String kafkaBrokers, String topic, SparkSession spark) {
        this.kafkaBrokers = kafkaBrokers;
        this.topic = topic;
        this.spark = spark;
    }

    @Override
    public Dataset<Row> getData() {
        Dataset<Row> data = spark.read()
                .format("kafka")
                .option("kafka.bootstrap.servers", kafkaBrokers)
                .option("subscribe", topic)
                .load();

        System.out.println("Считанные данные из Kafka:");
        data.show();
        return data;
    }
}

// Класс для выбора типа источника данных
class DataSourceSelector {
    public static DataSource createDataSource(String sourceType, String filePath, String kafkaBrokers, String kafkaTopic, SparkSession spark) {
        switch (sourceType.toLowerCase()) {
            case "file":
                return new CsvFileDataSource(filePath, spark); // В качестве примера используется CsvFileDataSource
            case "kafka":
                return new KafkaDataSource(kafkaBrokers, kafkaTopic, spark);
            // Добавить другие типы источников данных здесь
            default:
                throw new IllegalArgumentException("Неподдерживаемый тип источника данных: " + sourceType);
        }
    }
}

// Интерфейс для метрики расстояния
interface DistanceMetric extends Serializable {
    double compute(Vector a, Vector b);
}

// Реализация евклидовой метрики
class EuclideanDistance implements DistanceMetric {
    @Override
    public double compute(Vector a, Vector b) {
        return Vectors.sqdist(a, b);
    }
}

// Реализация манхэттенской метрики
class ManhattanDistance implements DistanceMetric {
    @Override
    public double compute(Vector a, Vector b) {
        double sum = 0.0;
        for (int i = 0; i < a.size(); i++) {
            sum += Math.abs(a.apply(i) - b.apply(i));
        }
        return sum;
    }
}

// Класс для создания метрики расстояния
class DistanceMetricSelector {
    public static DistanceMetric createMetric(String metric) {
        switch (metric.toLowerCase()) {
            case "euclidean":
                return new EuclideanDistance();
            case "manhattan":
                return new ManhattanDistance();
            // Добавить другие метрики здесь
            default:
                throw new IllegalArgumentException("Неподдерживаемая метрика расстояния: " + metric);
        }
    }
}

// Класс для обнаружения аномалий
class AnomalyDetector<T extends Model<? extends T>> implements Serializable {
    private final T clusteringModel;
    private final ResultWriter resultWriter;
    private final DistanceMetric distanceMetric;
    private final String[] selectedFeatures;

    public AnomalyDetector(T clusteringModel, ResultWriter resultWriter, DistanceMetric distanceMetric, String[] selectedFeatures) {
        this.clusteringModel = clusteringModel;
        this.resultWriter = resultWriter;
        this.distanceMetric = distanceMetric;
        this.selectedFeatures = selectedFeatures;
    }

    public boolean detectAnomalies(Dataset<Row> newData) {
        if (clusteringModel == null) {
            throw new IllegalStateException("Модель кластеризации еще не обучена.");
        }

        // Препроцессинг новых данных
        Dataset<Row> indexedData = AnomalyDetectorAppAdmin.preprocessData(newData, selectedFeatures);

        // Преобразование данных для кластеризации
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(indexedData.columns())
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(indexedData);

        // Обработка аномалий в зависимости от типа модели кластеризации
        if (clusteringModel instanceof KMeansModel) {
            return detectAnomalies((KMeansModel) clusteringModel, featureData, newData);
        } else if (clusteringModel instanceof BisectingKMeansModel) {
            return detectAnomalies((BisectingKMeansModel) clusteringModel, featureData, newData);
        } else {
            throw new IllegalArgumentException("Неподдерживаемый тип модели кластеризации.");
        }
    }

    // Обнаружение аномалий с помощью среднего и стандартного отклонения
    private boolean detectAnomalies(KMeansModel kmeansModel, Dataset<Row> indexedData, Dataset<Row> originalData) {
        Vector[] centers = kmeansModel.clusterCenters();
        ResultWriter resultWriter = this.resultWriter;

        List<Row> originalDataList = originalData.collectAsList();
        List<Row> indexedDataList = indexedData.collectAsList();

        boolean anomaliesDetected = false;
        List<Double> distances = new ArrayList<>();

        // Сбор всех расстояний
        for (int i = 0; i < indexedDataList.size(); i++) {
            Row row = indexedDataList.get(i);
            Vector features = row.getAs("features");
            double minDistance = Double.MAX_VALUE;
            for (Vector center : centers) {
                double distance = distanceMetric.compute(features, center);  // Вычисление расстояния с использованием выбранной метрики
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            distances.add(minDistance);
        }

        // Вычисление среднего и стандартного отклонения
        DescriptiveStatistics stats = new DescriptiveStatistics();
        distances.forEach(stats::addValue);
        double meanDistance = stats.getMean();
        double stdDevDistance = stats.getStandardDeviation();

        // Установка порога аномалии (например, среднее + 2 стандартных отклонения)
        double anomalyThreshold = meanDistance + 3 * stdDevDistance;

        // Обнаружение аномалий
        for (int i = 0; i < indexedDataList.size(); i++) {
            Row row = indexedDataList.get(i);
            Vector features = row.getAs("features");
            double minDistance = Double.MAX_VALUE;
            for (Vector center : centers) {
                double distance = distanceMetric.compute(features, center);
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            if (minDistance > anomalyThreshold) {
                String anomalyMessage = "Обнаружена аномалия в строке " + (originalDataList.get(i).getInt(0) + 1) + ": " + originalDataList.get(i);
                resultWriter.writeResult(anomalyMessage);
                System.out.println(anomalyMessage);
                anomaliesDetected = true;
            }
        }

        if (!anomaliesDetected) {
            System.out.println("Аномалии не обнаружены.");
        }
        return anomaliesDetected;
    }

    /*private boolean detectAnomalies(KMeansModel kmeansModel, Dataset<Row> indexedData, Dataset<Row> originalData) {
        Vector[] centers = kmeansModel.clusterCenters();
        ResultWriter resultWriter = this.resultWriter;

        List<Row> originalDataList = originalData.collectAsList();
        List<Row> indexedDataList = indexedData.collectAsList();

        boolean anomaliesDetected = false;

        for (int i = 0; i < indexedDataList.size(); i++) {
            Row row = indexedDataList.get(i);
            Vector features = row.getAs("features");
            double minDistance = Double.MAX_VALUE;
            for (Vector center : centers) {
                double distance = distanceMetric.compute(features, center);  // Вычисление расстояния с использованием выбранной метрики
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            double anomalyThreshold = 2 * Math.sqrt(minDistance);  // Вычисление порога аномалии
            if (minDistance > anomalyThreshold) {
                String anomalyMessage = "Обнаружена аномалия в строке " + (originalDataList.get(i).getInt(0) + 1) + ": " + originalDataList.get(i);
                resultWriter.writeResult(anomalyMessage);
                System.out.println(anomalyMessage);
                anomaliesDetected = true;
            }
        }

        if (!anomaliesDetected) {
            System.out.println("Аномалии не обнаружены.");
        }
        return anomaliesDetected;
    }*/

    private boolean detectAnomalies(BisectingKMeansModel bkmModel, Dataset<Row> indexedData, Dataset<Row> originalData) {
        Vector[] centers = bkmModel.clusterCenters();
        ResultWriter resultWriter = this.resultWriter;

        List<Row> originalDataList = originalData.collectAsList();
        List<Row> indexedDataList = indexedData.collectAsList();

        boolean anomaliesDetected = false;
        List<Double> distances = new ArrayList<>();

        // Сбор всех расстояний
        for (int i = 0; i < indexedDataList.size(); i++) {
            Row row = indexedDataList.get(i);
            Vector features = row.getAs("features");
            double minDistance = Double.MAX_VALUE;
            for (Vector center : centers) {
                double distance = distanceMetric.compute(features, center);  // Вычисление расстояния с использованием выбранной метрики
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            distances.add(minDistance);
        }

        // Вычисление среднего и стандартного отклонения
        DescriptiveStatistics stats = new DescriptiveStatistics();
        distances.forEach(stats::addValue);
        double meanDistance = stats.getMean();
        double stdDevDistance = stats.getStandardDeviation();

        // Установка порога аномалии (например, среднее + 2 стандартных отклонения)
        double anomalyThreshold = meanDistance + 3 * stdDevDistance;

        // Обнаружение аномалий
        for (int i = 0; i < indexedDataList.size(); i++) {
            Row row = indexedDataList.get(i);
            Vector features = row.getAs("features");
            double minDistance = Double.MAX_VALUE;
            for (Vector center : centers) {
                double distance = distanceMetric.compute(features, center);
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            if (minDistance > anomalyThreshold) {
                String anomalyMessage = "Обнаружена аномалия в строке " + (originalDataList.get(i).getInt(0) + 1) + ": " + originalDataList.get(i);
                resultWriter.writeResult(anomalyMessage);
                System.out.println(anomalyMessage);
                anomaliesDetected = true;
            }
        }

        if (!anomaliesDetected) {
            System.out.println("Аномалии не обнаружены.");
        }
        return anomaliesDetected;
    }

    /*private boolean detectAnomalies(BisectingKMeansModel bkmModel, Dataset<Row> indexedData, Dataset<Row> originalData) {
        Vector[] centers = bkmModel.clusterCenters();
        ResultWriter resultWriter = this.resultWriter;

        List<Row> originalDataList = originalData.collectAsList();
        List<Row> indexedDataList = indexedData.collectAsList();

        boolean anomaliesDetected = false;

        for (int i = 0; i < indexedDataList.size(); i++) {
            Row row = indexedDataList.get(i);
            Vector features = row.getAs("features");
            double minDistance = Double.MAX_VALUE;
            for (Vector center : centers) {
                double distance = distanceMetric.compute(features, center);  // Вычисление расстояния с использованием выбранной метрики
                if (distance < minDistance) {
                    minDistance = distance;
                }
            }
            double anomalyThreshold = 2 * Math.sqrt(minDistance);  // Вычисление порога аномалии
            if (minDistance > anomalyThreshold) {
                String anomalyMessage = "Обнаружена аномалия в строке " + (originalDataList.get(i).getInt(0) + 1) + ": " + originalDataList.get(i);
                resultWriter.writeResult(anomalyMessage);
                System.out.println(anomalyMessage);
                anomaliesDetected = true;
            }
        }

        if (!anomaliesDetected) {
            System.out.println("Аномалии не обнаружены.");
        }
        return anomaliesDetected;
    }*/
}

// Интерфейс для записи результатов
interface ResultWriter extends Serializable {
    void writeResult(String result);
}

// Класс для записи результатов в файл
class FileResultWriter implements ResultWriter {
    private final String filePath;

    public FileResultWriter(String filePath) {
        this.filePath = filePath;
    }

    @Override
    public void writeResult(String result) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
            writer.write(result);
            writer.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

// Класс для записи результатов в Kafka
class KafkaResultWriter implements ResultWriter {
    private final KafkaProducer<String, String> producer;
    private final String topic;

    public KafkaResultWriter(String brokers, String topic) {
        Properties props = new Properties();
        props.put("bootstrap.servers", brokers);
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        this.producer = new KafkaProducer<>(props);
        this.topic = topic;
    }

    @Override
    public void writeResult(String result) {
        producer.send(new ProducerRecord<>(topic, result));
    }
}

// Класс для выбора и создания ResultWriter
class ResultWriterSelector {
    public static ResultWriter createResultWriter(String outputType, String outputDestination, String kafkaBrokers, String kafkaTopic) {
        if (outputType.equalsIgnoreCase("file")) {
            return new FileResultWriter(outputDestination);
        } else if (outputType.equalsIgnoreCase("kafka")) {
            return new KafkaResultWriter(kafkaBrokers, kafkaTopic);
        } else {
            throw new IllegalArgumentException("Unsupported output type.");
        }
    }
}

// Интерфейс для кластеризации
interface Clusterizer<T extends Model<?>> {
    T cluster(Dataset<Row> data, String[] selectedFeatures);
}

// Класс для кластеризации с использованием алгоритма KMeans
class KMeansClusterizer implements Clusterizer<KMeansModel> {
    @Override
    public KMeansModel cluster(Dataset<Row> data, String[] selectedFeatures) {

        // Препроцессинг данных
        Dataset<Row> indexedData = AnomalyDetectorAppAdmin.preprocessData(data, selectedFeatures);

        // Определение оптимального числа кластеров
        int optimalClusters = determineOptimalClusters(indexedData);
        System.out.println("Оптимальное количество кластеров для KMeans: " + optimalClusters);

        // Преобразование данных для кластеризации
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(indexedData.columns())
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(indexedData);
        System.out.println("Данные после обработки для кластеризации:");
        featureData.show();

        // Создание и обучение модели KMeans
        KMeans kmeans = new KMeans()
                .setK(optimalClusters)
                .setSeed(1L)
                .setFeaturesCol("features")
                .setPredictionCol("prediction");

        KMeansModel model = kmeans.fit(featureData);

        // Вывод результатов кластеризации
        Dataset<Row> predictions = model.transform(featureData);
        System.out.println("Результаты кластеризации с KMeans:");
        predictions.show();

        return model;
    }

    // Метод для определения оптимального числа кластеров
    private int determineOptimalClusters(Dataset<Row> indexedData) {
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(indexedData.columns())
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(indexedData);

        // Определение оптимального числа кластеров с помощью метрики силуэта
        int minClusters = 3;
        int maxClusters = 10;
        int optimalClusters = 0;
        double maxSilhouette = 0;

        for (int k = minClusters; k <= maxClusters; k++) {
            KMeans kmeans = new KMeans()
                    .setK(k)
                    .setSeed(1L)
                    .setFeaturesCol("features")
                    .setPredictionCol("cluster");

            KMeansModel model = kmeans.fit(featureData);

            Dataset<Row> predictions = model.transform(featureData);

            ClusteringEvaluator evaluator = new ClusteringEvaluator()
                    .setFeaturesCol("features")
                    .setPredictionCol("cluster")
                    .setMetricName("silhouette");

            double silhouette = evaluator.evaluate(predictions);
            System.out.println("Значение силуэта для KMeans с " + k + " кластерами: " + silhouette);
            if (silhouette > maxSilhouette) {
                maxSilhouette = silhouette;
                optimalClusters = k;
            }
        }

        return optimalClusters;
    }
}

// Класс для кластеризации с использованием алгоритма Bisecting KMeans
class BisectingKMeansClusterizer implements Clusterizer<BisectingKMeansModel> {
    @Override
    public BisectingKMeansModel cluster(Dataset<Row> data, String[] selectedFeatures) {

        // Препроцессинг данных
        Dataset<Row> indexedData = AnomalyDetectorAppAdmin.preprocessData(data, selectedFeatures);

        int optimalClusters = determineOptimalClusters(indexedData);
        System.out.println("Оптимальное количество кластеров для Bisecting KMeans: " + optimalClusters);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(indexedData.columns())
                .setOutputCol("features");

        System.out.println("Содержимое indexedData после индексации и удаления строковых столбцов:");
        indexedData.show();
        indexedData.printSchema();

        Dataset<Row> featureData = assembler.transform(indexedData);
        System.out.println("Данные после обработки для кластеризации:");
        featureData.show();

        BisectingKMeans bkm = new BisectingKMeans()
                .setK(optimalClusters)
                .setSeed(1L)
                .setFeaturesCol("features")
                .setPredictionCol("prediction");

        BisectingKMeansModel model = bkm.fit(featureData);

        Dataset<Row> predictions = model.transform(featureData);
        System.out.println("Результаты кластеризации с Bisecting KMeans:");
        predictions.show();

        return model;
    }

    private int determineOptimalClusters(Dataset<Row> indexedData) {
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(indexedData.columns())
                .setOutputCol("features");

        Dataset<Row> featureData = assembler.transform(indexedData);

        // Определение оптимального числа кластеров с помощью метрики силуэта
        int minClusters = 3;
        int maxClusters = 10;
        int optimalClusters = 0;
        double maxSilhouette = 0;

        for (int k = minClusters; k <= maxClusters; k++) {
            BisectingKMeans bkm = new BisectingKMeans()
                    .setK(k)
                    .setSeed(1L)
                    .setFeaturesCol("features")
                    .setPredictionCol("cluster");

            BisectingKMeansModel model = bkm.fit(featureData);

            Dataset<Row> predictions = model.transform(featureData);

            ClusteringEvaluator evaluator = new ClusteringEvaluator()
                    .setFeaturesCol("features")
                    .setPredictionCol("cluster")
                    .setMetricName("silhouette");

            double silhouette = evaluator.evaluate(predictions);
            System.out.println("Значение силуэта для Bisecting KMeans с " + k + " кластерами: " + silhouette);
            if (silhouette > maxSilhouette) {
                maxSilhouette = silhouette;
                optimalClusters = k;
            }
        }

        return optimalClusters;
    }
}

// Выбор и создание кластеризатора
class ClusterizerSelector {
    public Clusterizer<?> createClusterizer(String algorithm) {
        if (algorithm.equalsIgnoreCase("kMeans")) {
            return new KMeansClusterizer();
        } else if (algorithm.equalsIgnoreCase("bisectingKMeans")) {
            return new BisectingKMeansClusterizer();
        } else {
            throw new IllegalArgumentException("Выбран неподдерживаемый алгоритм кластеризации.");
        }
    }
}


