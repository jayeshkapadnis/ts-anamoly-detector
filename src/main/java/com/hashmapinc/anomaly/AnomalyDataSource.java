package com.hashmapinc.anomaly;

import com.google.common.io.Files;
import com.google.common.io.LineProcessor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class AnomalyDataSource {

    private final String fileName;
    private final String separator;
    private final int seqLength;

    public DataSet train;
    public DataSet test;

    public AnomalyDataSource(String fileName, String separator, int seqLength) {
        this.fileName = fileName;
        this.separator = separator;
        this.seqLength = seqLength;
    }

    /**
     * use
     * @throws Exception
     */
    public void loadData() throws Exception {
        File data = new File(getClass().getClassLoader().getResource(fileName).toURI());
        int seq = seqLength + 1;
        double[][][] sequences = getWindowSequences(data, seq);

        INDArray dataSequences = Nd4j.create(sequences);

        DataSet dataset = new DataSet(dataSequences, dataSequences);

        dataset.shuffle();
        SplitTestAndTrain splitted = dataset.splitTestAndTrain(0.9);

        this.train = splitted.getTrain();
        this.test = splitted.getTest();
    }

    private double[][][] getWindowSequences(File data, int seq) throws IOException {
        List<List<Double>> result = readDataSet(data);
        double[][][] sequences = new double[result.size() - seq][seq][];
        for(int i =0; i < result.size() - seq; i++){
            List<List<Double>> lists = result.subList(i, seq + i);
            for (int j = 0; j < lists.size(); j++){
                //System.out.println(lists.get(j));
                sequences[i][j] = ArrayUtil.toArrayDouble(lists.get(j));
            }
        }
        return sequences;
    }

    private List<List<Double>> readDataSet(File data) throws IOException {
        return Files.readLines(data, Charset.defaultCharset(), new LineProcessor<List<List<Double>>>() {
            List<List<Double>> data = new ArrayList<>();

            public boolean processLine(String s) throws IOException {
                String[] split = s.split(separator);
                data.add(Arrays.stream(split)
                        .filter(c -> c != null && !c.isEmpty())
                        .map(Double::parseDouble).collect(Collectors.toList()));
                return true;
            }

            public List<List<Double>> getResult() {
                return data;
            }
        });
    }
}
