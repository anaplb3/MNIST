import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.io.File;

public class MNIST {

    public static void main(String[] args) throws Exception {
        SameDiff sd = SameDiff.create();

        // Propriedades do dataset
        int input = 28*28;
        int output = 10;

        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, input);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, output);

        // Definindo a camada
        int layerSize0 = 128;
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', input, layerSize0), DataType.FLOAT, input, layerSize0);
        SDVariable b0 = sd.zero("b0", 1, layerSize0);
        SDVariable activations0 = sd.nn().tanh(in.mmul(w0).add(b0));

        // Definindo a camada de output
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', layerSize0, output), DataType.FLOAT, layerSize0, output);
        SDVariable b1 = sd.zero("b1", 1, output);

        SDVariable z1 = activations0.mmul(w1).add("prediction", b1);
        SDVariable softmax = sd.nn().softmax("softmax", z1);

        // Definindo a loss function
        SDVariable diff = sd.math.squaredDifference(softmax, label);
        SDVariable lossMse = diff.mean();
        sd.setLossVariables(lossMse);

        // Criando e setando a configuração de treinamento
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .l2(1e-4)                               //L2 regularization
                .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
                .build();

        sd.setTrainingConfig(config);

        int batchSize = 32;
        DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator testData = new MnistDataSetIterator(batchSize, false, 12345);

        // Treino para duas epochs
        int numEpochs = 2;
        sd.fit(trainData, numEpochs);

        // Avaliando no set de testes
        String outputVariable = "softmax";
        Evaluation evaluation = new Evaluation();
        sd.evaluate(testData, outputVariable, evaluation);

        // Printando as avaliações  Print evaluation statistics:
        System.out.println(evaluation.stats());

        //Save the trained network for inference - FlatBuffers format
        File saveFileForInference = new File("sameDiffExampleInference.fb");
        sd.asFlatFile(saveFileForInference);

        SameDiff loadedForInference = SameDiff.fromFlatFile(saveFileForInference);

        //Perform inference on restored network
        INDArray example = new MnistDataSetIterator(1, false, 12345).next().getFeatures();
        loadedForInference.getVariable("input").setArray(example);
        INDArray finalOutput = loadedForInference.getVariable("softmax").eval();

        System.out.println("-----------------------");
        System.out.println(example.reshape(28, 28));
        System.out.println("Output probabilities: " + finalOutput);
        System.out.println("Predicted class: " + finalOutput.argMax().getInt(0));
    }
}
