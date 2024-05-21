import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.controlsfx.control.CheckComboBox;

import java.io.File;
import java.util.List;

public class AnomalyDetectorAppAdminGUI extends Application {

    private SparkSession spark;
    private String selectedFileType;
    private String selectedAlgorithm;
    private String selectedAnomalyMetric;
    private File selectedTrainFile;
    private File selectedTestFile;
    private CheckComboBox<String> featuresComboBox;
    private List<String> selectedFeatures;
    private Model<?> clusteringModel;
    private String[] featureNames;
    private ComboBox<String> sourceTypeComboBox;
    private ComboBox<String> fileTypeComboBox;
    private ComboBox<String> algorithmComboBox;
    private ComboBox<String> anomalyMetricComboBox;
    private ComboBox<String> resultOutputTypeComboBox;
    private TextField resultFilePathField;
    private TextField kafkaBrokersField;
    private TextField kafkaTopicField;
    private TextField kafkaInputBrokersField;
    private TextField kafkaInputTopicField;
    private TextField filePathField;
    private TextField testFilePathField;
    private Label fileLoadingStatusLabel;
    private Label trainingStatusLabel;
    private Label detectionStatusLabel;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        Logger.getLogger("org.apache.spark").setLevel(Level.ERROR);
        spark = SparkSession.builder()
                .appName("AnomalyDetectorAppAdminGUI")
                .config("spark.master", "local")
                .getOrCreate();

        primaryStage.setTitle("Anomaly Detector Admin");

        // Левая часть интерфейса
        Label sourceTypeLabel = new Label("Источник данных:");
        sourceTypeLabel.setStyle("-fx-font-size: 14;");
        sourceTypeComboBox = new ComboBox<>(FXCollections.observableArrayList("Файл", "Kafka"));
        sourceTypeComboBox.setStyle("-fx-font-size: 12;");
        sourceTypeComboBox.setPrefWidth(300);

        Label fileTypeLabel = new Label("Тип файла:");
        fileTypeLabel.setStyle("-fx-font-size: 14;");
        fileTypeComboBox = new ComboBox<>(FXCollections.observableArrayList("csv", "json"));
        fileTypeComboBox.setStyle("-fx-font-size: 12;");
        fileTypeComboBox.setDisable(true);
        fileTypeComboBox.setPrefWidth(300);

        Label selectFileLabel = new Label("Файл для обучения:");
        selectFileLabel.setStyle("-fx-font-size: 14;");
        filePathField = new TextField();
        filePathField.setPromptText("Выберите файл");
        filePathField.setStyle("-fx-font-size: 12;");
        filePathField.setEditable(false);
        filePathField.setDisable(true);
        filePathField.setPrefWidth(300);

        Button selectFileButton = new Button("Обзор");
        selectFileButton.setStyle("-fx-font-size: 14;");
        selectFileButton.setDisable(true);
        selectFileButton.setOnAction(e -> {
            FileChooser fileChooser = new FileChooser();
            selectedTrainFile = fileChooser.showOpenDialog(primaryStage);
            if (selectedTrainFile != null) {
                filePathField.setText(selectedTrainFile.getPath());
                loadFileFeatures(selectedTrainFile.getPath());
            }
        });

        fileLoadingStatusLabel = new Label();
        fileLoadingStatusLabel.setStyle("-fx-font-size: 12;");

        Label featuresLabel = new Label("Признаки:");
        featuresLabel.setStyle("-fx-font-size: 14;");
        featuresComboBox = new CheckComboBox<>();
        featuresComboBox.setStyle("-fx-font-size: 12;");
        featuresComboBox.setPrefWidth(300);
        featuresComboBox.setDisable(true);

        Label kafkaInputBrokersLabel = new Label("Настройки Kafka:");
        kafkaInputBrokersLabel.setStyle("-fx-font-size: 14;");
        kafkaInputBrokersField = new TextField();
        kafkaInputBrokersField.setStyle("-fx-font-size: 12;");
        kafkaInputBrokersField.setPromptText("Kafka Brokers");
        kafkaInputBrokersField.setDisable(true);
        kafkaInputBrokersField.setPrefWidth(100);

        kafkaInputTopicField = new TextField();
        kafkaInputTopicField.setStyle("-fx-font-size: 12;");
        kafkaInputTopicField.setPromptText("Kafka Topic");
        kafkaInputTopicField.setDisable(true);
        kafkaInputTopicField.setPrefWidth(100);

        Label algorithmLabel = new Label("Алгоритм:");
        algorithmLabel.setStyle("-fx-font-size: 14;");
        algorithmComboBox = new ComboBox<>(FXCollections.observableArrayList("KMeans", "BisectingKMeans"));
        algorithmComboBox.setStyle("-fx-font-size: 12;");
        algorithmComboBox.setPrefWidth(300);
        algorithmComboBox.valueProperty().addListener((observable, oldValue, newValue) -> selectedAlgorithm = newValue);

        Button startTrainingButton = new Button("Начать обучение");
        startTrainingButton.setStyle("-fx-font-size: 14;");
        trainingStatusLabel = new Label();
        trainingStatusLabel.setStyle("-fx-font-size: 12;");

        Label selectTestFileLabel = new Label("Файл для проверки:");
        selectTestFileLabel.setStyle("-fx-font-size: 14;");
        testFilePathField = new TextField();
        testFilePathField.setPromptText("Выберите файл");
        testFilePathField.setStyle("-fx-font-size: 12;");
        testFilePathField.setEditable(false);
        testFilePathField.setPrefWidth(300);

        Button selectTestFileButton = new Button("Обзор");
        selectTestFileButton.setStyle("-fx-font-size: 12;");
        selectTestFileButton.setOnAction(e -> {
            FileChooser fileChooser = new FileChooser();
            selectedTestFile = fileChooser.showOpenDialog(primaryStage);
            if (selectedTestFile != null) {
                testFilePathField.setText(selectedTestFile.getPath());
            }
        });

        Label anomalyMetricLabel = new Label("Метрика аномалий:");
        anomalyMetricLabel.setStyle("-fx-font-size: 14;");
        anomalyMetricComboBox = new ComboBox<>(FXCollections.observableArrayList("Manhattan", "Euclidean"));
        anomalyMetricComboBox.setStyle("-fx-font-size: 12;");
        anomalyMetricComboBox.setPrefWidth(300);
        anomalyMetricComboBox.valueProperty().addListener((observable, oldValue, newValue) -> selectedAnomalyMetric = newValue);

        Label resultOutputTypeLabel = new Label("Тип источника для результатов:");
        resultOutputTypeLabel.setStyle("-fx-font-size: 14;");
        resultOutputTypeComboBox = new ComboBox<>(FXCollections.observableArrayList("Файл", "Kafka"));
        resultOutputTypeComboBox.setStyle("-fx-font-size: 12;");
        resultOutputTypeComboBox.setPrefWidth(300);

        Label resultFilePathLabel = new Label("Сохранение файла:");
        resultFilePathLabel.setStyle("-fx-font-size: 14;");
        resultFilePathField = new TextField();
        resultFilePathField.setPromptText("Выберите путь");
        resultFilePathField.setStyle("-fx-font-size: 12;");
        resultFilePathField.setDisable(true);
        resultFilePathField.setPrefWidth(300);

        Button selectResultFileButton = new Button("Обзор");
        selectResultFileButton.setStyle("-fx-font-size: 12;");
        selectResultFileButton.setDisable(true);
        selectResultFileButton.setOnAction(e -> {
            FileChooser fileChooser = new FileChooser();
            File resultFile = fileChooser.showSaveDialog(primaryStage);
            if (resultFile != null) {
                resultFilePathField.setText(resultFile.getPath());
            }
        });

        Label kafkaSettingsLabel = new Label("Настройки Kafka:");
        kafkaSettingsLabel.setStyle("-fx-font-size: 14;");
        kafkaBrokersField = new TextField();
        kafkaBrokersField.setStyle("-fx-font-size: 12;");
        kafkaBrokersField.setPromptText("Kafka Brokers");
        kafkaBrokersField.setPrefWidth(300);
        kafkaTopicField = new TextField();
        kafkaTopicField.setStyle("-fx-font-size: 12;");
        kafkaTopicField.setPromptText("Kafka Topic");
        kafkaTopicField.setPrefWidth(300);
        kafkaBrokersField.setDisable(true);
        kafkaTopicField.setDisable(true);

        resultOutputTypeComboBox.valueProperty().addListener((observable, oldValue, newValue) -> {
            if ("Файл".equals(newValue)) {
                resultFilePathField.setDisable(false);
                selectResultFileButton.setDisable(false);
                kafkaBrokersField.setDisable(true);
                kafkaTopicField.setDisable(true);
            } else if ("Kafka".equals(newValue)) {
                resultFilePathField.setDisable(true);
                selectResultFileButton.setDisable(true);
                kafkaBrokersField.setDisable(false);
                kafkaTopicField.setDisable(false);
            }
        });

        sourceTypeComboBox.valueProperty().addListener((observable, oldValue, newValue) -> {
            if ("Файл".equals(newValue)) {
                fileTypeComboBox.setDisable(false);
                filePathField.setDisable(false);
                selectFileButton.setDisable(false);
                kafkaInputBrokersField.setDisable(true);
                kafkaInputTopicField.setDisable(true);
            } else if ("Kafka".equals(newValue)) {
                fileTypeComboBox.setDisable(true);
                filePathField.setDisable(true);
                selectFileButton.setDisable(true);
                kafkaInputBrokersField.setDisable(false);
                kafkaInputTopicField.setDisable(false);
            }
        });

        Button startAnomalyDetectionButton = new Button("Начать проверку");
        startAnomalyDetectionButton.setStyle("-fx-font-size: 14;");
        detectionStatusLabel = new Label();
        detectionStatusLabel.setStyle("-fx-font-size: 12;");

        Button clearButton = new Button("Очистить");
        clearButton.setStyle("-fx-font-size: 14;");
        clearButton.setOnAction(e -> clearAllFields());

        Button exitButton = new Button("Выход");
        exitButton.setStyle("-fx-font-size: 14;");
        exitButton.setOnAction(e -> {
            spark.stop();
            primaryStage.close();
        });

        // Левая часть
        VBox topVBox = new VBox(10);
        topVBox.setPadding(new Insets(30));
        topVBox.setSpacing(10);
        topVBox.getChildren().addAll(
                sourceTypeLabel, sourceTypeComboBox,
                new Label(""),
                fileTypeLabel, fileTypeComboBox,
                new Label(""),
                selectFileLabel, filePathField, selectFileButton,
                fileLoadingStatusLabel,
                featuresLabel, featuresComboBox,
                new Label(""),
                kafkaInputBrokersLabel, kafkaInputBrokersField, kafkaInputTopicField,
                new Label(""),
                algorithmLabel, algorithmComboBox,
                new Label(""),
                startTrainingButton, trainingStatusLabel
        );

        // Правая часть
        VBox bottomVBox = new VBox(10);
        bottomVBox.setPadding(new Insets(30));
        bottomVBox.setSpacing(10);
        bottomVBox.getChildren().addAll(
                selectTestFileLabel, testFilePathField, selectTestFileButton,
                new Label(""),
                anomalyMetricLabel, anomalyMetricComboBox,
                new Label(""),
                resultOutputTypeLabel, resultOutputTypeComboBox,
                new Label(""),
                resultFilePathLabel, resultFilePathField, selectResultFileButton,
                new Label(""),
                kafkaSettingsLabel, kafkaBrokersField, kafkaTopicField,
                new Label(""),
                startAnomalyDetectionButton, detectionStatusLabel
        );

        HBox buttonContainer = new HBox(10);
        buttonContainer.setAlignment(Pos.BOTTOM_RIGHT);
        buttonContainer.getChildren().addAll(clearButton, exitButton);

        bottomVBox.getChildren().add(buttonContainer);

        // Разделение интерфейса
        SplitPane splitPane = new SplitPane();
        splitPane.getItems().addAll(topVBox, bottomVBox);
        splitPane.setDividerPositions(0.5);

        Scene scene = new Scene(splitPane, 1200, 800);
        primaryStage.setScene(scene);
        primaryStage.setMaximized(true);
        primaryStage.show();

        startTrainingButton.setOnAction(e -> startTrainingButtonHandler());

        startAnomalyDetectionButton.setOnAction(e -> startAnomalyDetectionButtonHandler());
    }

    private void startTrainingButtonHandler() {
        if (validateTrainingInputs()) {
            Task<Void> trainingTask = new Task<Void>() {
                @Override
                protected Void call() {
                    updateMessage("Подождите, идет процесс обучения...");
                    try {
                        selectedFileType = fileTypeComboBox.getValue();
                        selectedAlgorithm = algorithmComboBox.getValue();
                        selectedFeatures = featuresComboBox.getCheckModel().getCheckedItems();
                        if (selectedTrainFile != null) {
                            String filePath = selectedTrainFile.getPath();
                            DataSource dataSource = FileDataSourceSelector.createFileDataSource(selectedFileType, filePath, spark);
                            Dataset<Row> data = dataSource.getData();

                            ClusterizerSelector clusterizerSelect = new ClusterizerSelector();
                            Clusterizer<?> clusterizer = clusterizerSelect.createClusterizer(selectedAlgorithm);
                            clusteringModel = clusterizer.cluster(data, selectedFeatures.toArray(new String[0]));
                            updateMessage("Модель обучена");

                            // Добавление отладочного сообщения
                            System.out.println("Модель обучена успешно.");
                        } else {
                            updateMessage("Пожалуйста, выберите файл");
                            System.out.println("Файл не выбран.");
                        }
                    } catch (Exception ex) {
                        updateMessage("Ошибка при обучении модели: " + ex.getMessage());
                        // Добавление отладочного сообщения
                        System.err.println("Error during model training: " + ex.getMessage());
                    }
                    return null;
                }
            };

            trainingStatusLabel.textProperty().bind(trainingTask.messageProperty());
            new Thread(trainingTask).start();
        } else {
            trainingStatusLabel.setText("Заполните все обязательные поля!");
            System.out.println("Заполнены не все поля для обучения.");
        }
    }

    private void startAnomalyDetectionButtonHandler() {
        if (validateDetectionInputs()) {
            Task<Void> detectionTask = new Task<Void>() {
                @Override
                protected Void call() {
                    updateMessage("Подождите, идет процесс проверки...");
                    try {
                        selectedAnomalyMetric = anomalyMetricComboBox.getValue();
                        if (selectedTestFile != null && clusteringModel != null) {
                            String filePath = selectedTestFile.getPath();
                            DataSource dataSource = FileDataSourceSelector.createFileDataSource(selectedFileType, filePath, spark);
                            Dataset<Row> newData = dataSource.getData();

                            // Создание ResultWriter на основе выбранного источника записи результатов проверки
                            ResultWriter resultWriter;
                            if (resultOutputTypeComboBox.getValue().equals("Файл")) {
                                String resultFilePath = resultFilePathField.getText();
                                resultWriter = new FileResultWriter(resultFilePath);
                            } else {
                                String brokers = kafkaBrokersField.getText();
                                String topic = kafkaTopicField.getText();
                                resultWriter = new KafkaResultWriter(brokers, topic);
                            }

                            // Выбор метрики расстояния
                            DistanceMetric distanceMetric = DistanceMetricSelector.createMetric(selectedAnomalyMetric.toLowerCase());

                            // Создание экземпляра класса AnomalyDetector
                            AnomalyDetector<?> anomalyDetector;
                            if (clusteringModel instanceof KMeansModel) {
                                anomalyDetector = new AnomalyDetector<>((KMeansModel) clusteringModel, resultWriter, distanceMetric, selectedFeatures.toArray(new String[0]));
                            } else if (clusteringModel instanceof BisectingKMeansModel) {
                                anomalyDetector = new AnomalyDetector<>((BisectingKMeansModel) clusteringModel, resultWriter, distanceMetric, selectedFeatures.toArray(new String[0]));
                            } else {
                                throw new IllegalArgumentException("Неподдерживаемый алгоритм");
                            }

                            // Обработка и обнаружение аномалий на новых данных
                            boolean anomaliesDetected = anomalyDetector.detectAnomalies(newData);
                            if (anomaliesDetected) {
                                updateMessage("ОБНАРУЖЕНЫ АНОМАЛИИ В ДАННЫХ!!! Проверьте результаты выполнения обработки.");
                            } else {
                                updateMessage("Аномалии в данных не обнаружены.");
                            }
                        } else {
                            updateMessage("Пожалуйста, выберите файл для проверки и обучите модель");
                        }
                    } catch (Exception ex) {
                        updateMessage("Ошибка при проверке: " + ex.getMessage());
                        System.err.println("Ошибка в процессе обнаружения аномалий: " + ex.getMessage());
                    }
                    return null;
                }
            };

            detectionStatusLabel.textProperty().bind(detectionTask.messageProperty());
            new Thread(detectionTask).start();
        } else {
            detectionStatusLabel.setText("Заполните все обязательные поля!");
        }
    }

    private void loadFileFeatures(String filePath) {
        Task<Void> loadTask = new Task<Void>() {
            @Override
            protected Void call() throws Exception {
                updateMessage("Загрузка признаков из файла...");
                try {
                    Dataset<Row> data = spark.read()
                            .format("csv")
                            .option("header", "true")
                            .load(filePath);

                    featureNames = data.columns();
                    updateMessage("Загрузка завершена. Выберите признаки");

                    // Добавление отладочного сообщения
                    System.out.println("Загрузка признаков из файла: " + String.join(", ", featureNames));
                } catch (Exception e) {
                    updateMessage("Ошибка при загрузке файла.");
                    // Добавление отладочного сообщения
                    System.err.println("Ошибка при загрузке файла: " + e.getMessage());
                }
                return null;
            }
        };

        fileLoadingStatusLabel.textProperty().bind(loadTask.messageProperty());

        loadTask.setOnSucceeded(e -> {
            featuresComboBox.getItems().clear();  // Очищаем селектор перед добавлением новых признаков
            featuresComboBox.getItems().addAll(featureNames);
            featuresComboBox.setDisable(false);

            // Добавление отладочного сообщения
            System.out.println("Селектор с признаками обновлен.");
        });

        Thread loadThread = new Thread(loadTask);
        loadThread.setDaemon(true);
        loadThread.start();
    }

    private void clearAllFields() {
        // Сброс всех полей и селекторов
        sourceTypeComboBox.setValue(null);
        fileTypeComboBox.setValue(null);
        algorithmComboBox.setValue(null);
        anomalyMetricComboBox.setValue(null);
        resultOutputTypeComboBox.setValue(null);

        filePathField.clear();
        testFilePathField.clear();
        resultFilePathField.clear();
        kafkaBrokersField.clear();
        kafkaTopicField.clear();
        kafkaInputBrokersField.clear();
        kafkaInputTopicField.clear();

        // Очистка и деактивация селектора признаков
        featuresComboBox.getItems().clear();
        featuresComboBox.setDisable(true);

        // Очистка сообщений статуса
        fileLoadingStatusLabel.textProperty().unbind();
        fileLoadingStatusLabel.setText("");
        trainingStatusLabel.textProperty().unbind();
        trainingStatusLabel.setText("");
        detectionStatusLabel.textProperty().unbind();
        detectionStatusLabel.setText("");

        // Сброс переменных
        selectedFileType = null;
        selectedAlgorithm = null;
        selectedAnomalyMetric = null;
        selectedTrainFile = null;
        selectedTestFile = null;
        selectedFeatures = null;
        featureNames = null;
        clusteringModel = null;

        // Добавление отладочных сообщений
        System.out.println("Поля очищены.");
    }

    private boolean validateTrainingInputs() {
        boolean valid = true;

        System.out.println("Validation state at startTrainingButton click:");
        System.out.println("selectedTrainFile: " + (selectedTrainFile != null ? selectedTrainFile.getPath() : "null"));
        System.out.println("featuresComboBox disabled: " + featuresComboBox.isDisabled());
        System.out.println("Checked features: " + featuresComboBox.getCheckModel().getCheckedItems());
        System.out.println("selectedAlgorithm: " + selectedAlgorithm);
        System.out.println("kafkaInputBrokersField disabled: " + kafkaInputBrokersField.isDisabled());
        System.out.println("kafkaInputBrokersField: " + kafkaInputBrokersField.getText());
        System.out.println("kafkaInputTopicField disabled: " + kafkaInputTopicField.isDisabled());
        System.out.println("kafkaInputTopicField: " + kafkaInputTopicField.getText());

        if (selectedTrainFile == null) valid = false;
        if (!featuresComboBox.isDisabled() && featuresComboBox.getCheckModel().getCheckedItems().isEmpty())
            valid = false;
        if (selectedAlgorithm == null || selectedAlgorithm.isEmpty()) valid = false;
        if (!kafkaInputBrokersField.isDisabled() && kafkaInputBrokersField.getText().isEmpty()) valid = false;
        if (!kafkaInputTopicField.isDisabled() && kafkaInputTopicField.getText().isEmpty()) valid = false;

        // Добавление отладочного сообщения
        System.out.println("Результат валидации полей для обучения: " + valid);

        return valid;
    }

    private boolean validateDetectionInputs() {
        boolean valid = true;

        System.out.println("Validation state at startAnomalyDetectionButton click:");
        System.out.println("selectedTestFile: " + (selectedTestFile != null ? selectedTestFile.getPath() : "null"));
        System.out.println("selectedAnomalyMetric: " + selectedAnomalyMetric);
        System.out.println("clusteringModel: " + (clusteringModel != null ? "exists" : "null"));
        System.out.println("resultFilePathField disabled: " + resultFilePathField.isDisabled());
        System.out.println("resultFilePathField: " + resultFilePathField.getText());
        System.out.println("kafkaBrokersField disabled: " + kafkaBrokersField.isDisabled());
        System.out.println("kafkaBrokersField: " + kafkaBrokersField.getText());
        System.out.println("kafkaTopicField disabled: " + kafkaTopicField.isDisabled());
        System.out.println("kafkaTopicField: " + kafkaTopicField.getText());

        if (selectedTestFile == null) valid = false;
        if (selectedAnomalyMetric == null || selectedAnomalyMetric.isEmpty()) valid = false;
        if (clusteringModel == null) valid = false;
        if (!resultFilePathField.isDisabled() && resultFilePathField.getText().isEmpty()) valid = false;
        if (!kafkaBrokersField.isDisabled() && kafkaBrokersField.getText().isEmpty()) valid = false;
        if (!kafkaTopicField.isDisabled() && kafkaTopicField.getText().isEmpty()) valid = false;

        // Добавление отладочного сообщения
        System.out.println("Результат валидации полей для проверки на аномалии: " + valid);

        return valid;
    }
}
