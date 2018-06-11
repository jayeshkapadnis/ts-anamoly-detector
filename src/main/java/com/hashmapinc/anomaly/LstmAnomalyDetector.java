package com.hashmapinc.anomaly;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.ImmutablePair;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class LstmAnomalyDetector {

    private final AnomalyDataSource source;
    private final String modelPath;

    private double threshold = 0.01;

    public LstmAnomalyDetector(AnomalyDataSource source, String pathToModel) {
        this.source = source;
        this.modelPath = pathToModel;
    }

    private List<INDArray> features(DataSet d){
        return d.asList().stream().map(DataSet::getFeatureMatrix).collect(Collectors.toList());
    }

    public MultiLayerNetwork buildModel(int[] layers){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123456)
                .optimizationAlgo( OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(2e-3))
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(0, new LSTM.Builder().name("input").nIn(layers[0]).nOut(layers[1]).build())
                .layer(1, new LSTM.Builder().name("encoder1").nIn(layers[1]).nOut(layers[2]).build())
                .layer(2, new LSTM.Builder().name("encoder2").nIn(layers[2]).nOut(layers[3]).build())
                .layer(3, new LSTM.Builder().name("decoder1").nIn(layers[3]).nOut(layers[2]).build())
                .layer(4, new LSTM.Builder().name("decoder2").nIn(layers[2]).nOut(layers[1]).build())
                .layer(5, new RnnOutputLayer.Builder().name("output").nIn(layers[1]).nOut(layers[0])
                        .activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build()) //MSE is sensitive to outliers so good for anamoly detection
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

    public void train(int nEpochs, int batchSize) throws Exception {
        source.loadData();
        int[] layers = new int[]{source.train.getFeatures().shape()[1], 200, 100, 10};
        MultiLayerNetwork net = buildModel(layers);
        StatsStorage statsStorage = new InMemoryStatsStorage();
        net.setListeners(new StatsListener(statsStorage),new ScoreIterationListener(2));

        //fitTrainData(nEpochs, net);

        fitTrainDataWithIterator(nEpochs, batchSize, net);

        List<ImmutablePair<Double, INDArray>> evalList = testScores(net);

        Collections.sort(evalList, new PairComparator());
        List<INDArray> normalList = new ArrayList<>();
        List<INDArray> anomalyList = new ArrayList<>();

        int listsize = evalList.size();
        for( int j = 0; j < listsize && j < 60; j ++ ){
            normalList.add(evalList.get(j).getRight());
            anomalyList.add(evalList.get(listsize -j -1).getRight());
        }

        System.out.println("Best and worst "+ normalList.size() + " " + anomalyList.size());

        File modelFile = new File(modelPath + "/AbnormalDetectedModel.zip");
        if (!modelFile.exists()) {
            modelFile.createNewFile();
        }

        ModelSerializer.writeModel(net, modelFile,true);

    }

    private void fitTrainDataWithIterator(int nEpochs, int batchSize, MultiLayerNetwork net) {
        ListDataSetIterator<DataSet> iterator = new ListDataSetIterator<>(source.train.asList(), batchSize);

        for (int i = 0; i < nEpochs; i++) {
            net.fit(iterator);
            System.out.println("Epoch " + i + " complete");
        }
    }

    private void fitTrainData(int nEpochs, MultiLayerNetwork net) {
        for (int i = 0; i < nEpochs; i++) {
            List<DataSet> dataSets = source.train.asList();
            System.out.println("No of datasets "+dataSets.size());
            for (DataSet dataSet : dataSets) {
                net.fit(dataSet);
            }
            System.out.println("Epoch " + i + " complete");
        }
    }

    @NotNull
    private List<ImmutablePair<Double, INDArray>> testScores(MultiLayerNetwork net) {
        List<DataSet> testDatasets = source.test.asList();

        List<ImmutablePair<Double,INDArray>> evalList = new ArrayList<>();

        int totalNum = testDatasets.size();
        double totalScore = 0;
        for( int i = 0; i < testDatasets.size(); i ++ ){
            INDArray testData = testDatasets.get(i).getFeatureMatrix();
            double score = net.score(new DataSet(testData, testData));
            totalScore += score;
            evalList.add(new ImmutablePair<>(score, testData));
            System.out.println("featuresTest " + i + " complete");
        }
        return evalList;
    }

    private static class PairComparator implements Comparator<ImmutablePair<Double, INDArray>>{

        @Override
        public int compare(ImmutablePair<Double, INDArray> o1, ImmutablePair<Double, INDArray> o2) {
            return Double.compare(o1.getLeft(),o2.getLeft());
        }
    }
}
