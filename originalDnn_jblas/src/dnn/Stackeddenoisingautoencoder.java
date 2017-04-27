package dnn;

import java.util.ArrayList;
import java.util.List;

import layers.Hiddenlayer;

import org.jblas.FloatMatrix;

import single_layer.Logistic_kaiki;
import util.Common_method;
import util.RandomGenerator;
import Mersenne.Sfmt;

public class Stackeddenoisingautoencoder {

	int input_N;
	int output_N;
	Sfmt mt;
	int[] hidden_N;//各隠れ層のノード数
	int hiddenlayer_N;//隠れ層の数
	Hiddenlayer[] sigmoidlayer;
	Denoisingautoencoder[] dae;
	Logistic_kaiki logisticlayer;

	public Stackeddenoisingautoencoder(int input, int[] hidden, int output, int minibatch,Sfmt m){
		if(m == null){
			int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
			m = new Sfmt(init_key);
		}

		input_N=input;
		output_N=output;
		hidden_N=hidden;
		mt=m;
		hiddenlayer_N=hidden.length;

		sigmoidlayer = new Hiddenlayer[hiddenlayer_N];
		dae = new Denoisingautoencoder[hiddenlayer_N];
		for(int i=0; i<hiddenlayer_N; i++){
			int num_input = 0;
			if(i==0){
				num_input = input;
			}else{
				num_input = hidden[i-1];
			}
			//隠れ層。シグモイドが活性化関数
			sigmoidlayer[i] = new Hiddenlayer(num_input, hidden[i], null, null, minibatch,m, "sigmoid");
			//デノイジングオートエンコーダを構成。重み、バイアスは隠れ層と共有
			dae[i] = new Denoisingautoencoder(num_input, hidden[i], sigmoidlayer[i].weight, sigmoidlayer[i].bias, null, mt);
		}
		//出力層。ロジスティック回帰
		logisticlayer = new Logistic_kaiki(hidden[hiddenlayer_N-1], output);
	}

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		Sfmt mt = new Sfmt(init_key);
		//
		// Declare variables and constants
		//

		int train_N_each = 200;        // for demo
		int validation_N_each = 200;   // for demo
		int test_N_each = 50;          // for demo
		int nIn_each = 20;             // for demo
		float pNoise_Training = 0.2f;  // for demo
		float pNoise_Test = 0.25f;     // for demo

		final int patterns = 3;

		final int train_N = train_N_each * patterns;
		final int validation_N = validation_N_each * patterns;
		final int test_N = test_N_each * patterns;

		final int nIn = nIn_each * patterns;
		final int nOut = patterns;
		int[] hiddenLayerSizes = {20, 20};
		float noise_lvl = 0.3f;

		FloatMatrix train_X = new FloatMatrix(train_N,nIn);

		FloatMatrix validation_X = new FloatMatrix(validation_N,nIn);  // type is set to float here, but exact values are int
		FloatMatrix validation_T = new FloatMatrix(validation_N,nOut);

		FloatMatrix test_X = new FloatMatrix(test_N,nIn);  // type is set to float here, but exact values are int
		FloatMatrix test_T = new FloatMatrix(test_N,nOut);
		FloatMatrix predicted_T = new FloatMatrix(test_N,nOut);

		int pretrainEpochs = 1000;
		float pretrainLearningRate = 0.2f;
		int finetuneEpochs = 1000;
		float finetuneLearningRate = 0.15f;

		int minibatchSize = 50;
		final int train_minibatch_N = train_N / minibatchSize;
		final int validation_minibatch_N = validation_N / minibatchSize;

		//
		// Create training data and test data for demo.
		//
		for (int pattern = 0; pattern < patterns; pattern++) {
			for (int n = 0; n < train_N_each; n++) {
				int n_ = pattern * train_N_each + n;
				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= train_N_each * pattern && n_ < train_N_each * (pattern + 1) ) &&
							(i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
						train_X.put(n_,i,(float) (RandomGenerator.binomial(1, 1 - pNoise_Training, mt) * mt.NextUnif() *0.5+0.5));
					} else {
						train_X.put(n_,i,(float)(RandomGenerator.binomial(1, pNoise_Training, mt) * mt.NextUnif() *0.5+0.5));
					}
				}
			}

			for (int n = 0; n < validation_N_each; n++) {
				int n_ = pattern * validation_N_each + n;
				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= validation_N_each * pattern && n_ < validation_N_each * (pattern + 1) ) &&
							(i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
						validation_X.put(n_,i, (float)(RandomGenerator.binomial(1, 1 - pNoise_Training, mt) * mt.NextUnif() *0.5+0.5));
					} else {
						validation_X.put(n_,i, (float)(RandomGenerator.binomial(1, pNoise_Training, mt) * mt.NextUnif() *0.5+0.5));
					}
				}

				for (int i = 0; i < nOut; i++) {
					if (i == pattern) {
						validation_T.put(n_,i,1);
					} else {
						validation_T.put(n_,i,0);
					}
				}
			}

			for (int n = 0; n < test_N_each; n++) {
				int n_ = pattern * test_N_each + n;
				for (int i = 0; i < nIn; i++) {
					if ( (n_ >= test_N_each * pattern && n_ < test_N_each * (pattern + 1) ) &&
							(i >= nIn_each * pattern && i < nIn_each * (pattern + 1)) ) {
						test_X.put(n_,i, (float)(RandomGenerator.binomial(1, 1 - pNoise_Test, mt) * mt.NextUnif() *0.5+0.5));
					} else {
						test_X.put(n_,i, (float)(RandomGenerator.binomial(1, pNoise_Test, mt) * mt.NextUnif() *0.5+0.5));
					}
				}

				for (int i = 0; i < nOut; i++) {
					if (i == pattern) {
						test_T.put(n_,i, 1);
					} else {
						test_T.put(n_,i, 0);
					}
				}
			}
		}


		FloatMatrix[] train_X_minibatch = new FloatMatrix[train_minibatch_N];//[minibatchSize][nIn];
		FloatMatrix[] validation_X_minibatch = new FloatMatrix[validation_minibatch_N];//[minibatchSize][nIn];
		FloatMatrix[] validation_T_minibatch = new FloatMatrix[validation_minibatch_N];//[minibatchSize][nOut];
		List<Integer> minibatchIndex = new ArrayList<>();
		for (int i = 0; i < train_N; i++) {minibatchIndex.add(i);}
		Common_method.list_shuffle(minibatchIndex, mt);

		for (int i = 0; i < train_minibatch_N; i++) {
			train_X_minibatch[i] = new FloatMatrix(minibatchSize,nIn);
		}
		for (int i = 0; i < validation_minibatch_N; i++) {
			validation_X_minibatch[i] = new FloatMatrix(minibatchSize,nIn);
			validation_T_minibatch[i] = new FloatMatrix(minibatchSize,nOut);
		}

		// create minibatches
		for (int j = 0; j < minibatchSize; j++) {
			for (int i = 0; i < train_minibatch_N; i++) {
				train_X_minibatch[i].putRow(j, train_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
			for (int i = 0; i < validation_minibatch_N; i++) {
				validation_X_minibatch[i].putRow(j,validation_X.getRow(minibatchIndex.get(i * minibatchSize + j)));
				validation_T_minibatch[i].putRow(j,validation_T.getRow(minibatchIndex.get(i * minibatchSize + j)));
			}
		}

		System.out.print("Building the model...");
		Stackeddenoisingautoencoder classifier = new Stackeddenoisingautoencoder(nIn, hiddenLayerSizes, nOut,minibatchSize, mt);
		System.out.println("done.");

		// pre-training the model
		System.out.print("Pre-training the model...");
		classifier.pretrain(train_X_minibatch, minibatchSize, train_minibatch_N, pretrainEpochs, pretrainLearningRate, noise_lvl);
		System.out.println("done.");

		System.out.println("Fine-tuning the model...");
		for(int epoch=0; epoch<finetuneEpochs; epoch++){
			for(int batch=0; batch<validation_minibatch_N; batch++){
				classifier.finetune(validation_X_minibatch[batch], validation_T_minibatch[batch], minibatchSize, finetuneLearningRate);
			}
			finetuneLearningRate *= 0.98;
		}

		// test
		predicted_T = classifier.predict(test_X);

		Common_method.print_result(predicted_T, test_T);
		//
		// Evaluate the model
		//

/*		int[][] confusionMatrix = new int[patterns][patterns];
		double accuracy = 0.;
		double[] precision = new double[patterns];
		double[] recall = new double[patterns];

		for (int i = 0; i < test_N; i++) {
			int predicted_ = Arrays.asList(predicted_T[i]).indexOf(1);
			int actual_ = Arrays.asList(test_T[i]).indexOf(1);

			confusionMatrix[actual_][predicted_] += 1;
		}

		for (int i = 0; i < patterns; i++) {
			double col_ = 0.;
			double row_ = 0.;

			for (int j = 0; j < patterns; j++) {

				if (i == j) {
					accuracy += confusionMatrix[i][j];
					precision[i] += confusionMatrix[j][i];
					recall[i] += confusionMatrix[i][j];
				}

				col_ += confusionMatrix[j][i];
				row_ += confusionMatrix[i][j];
			}
			precision[i] /= col_;
			recall[i] /= row_;
		}

		accuracy /= test_N;

		System.out.println("--------------------");
		System.out.println("SDA model evaluation");
		System.out.println("--------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, precision[i] * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, recall[i] * 100);
		}
*/
	}

	/**
	 * ファインチューニングメソッド
	 * @param x ミニバッチに分けたトレーニングデータ
	 * @param label ラベルデータ
	 * @param minibatchSize ミニバッチサイズ
	 * @param l_rate 学習率
	 */
	public void finetune(FloatMatrix x, FloatMatrix label, int minibatchSize, float l_rate) {
//		public void finetune(float[][] x, int[][] label, int minibatchSize, float l_rate) {
		// TODO 自動生成されたメソッド・スタブ
		List<FloatMatrix> inputdata = new ArrayList<>(hiddenlayer_N+1);
//		List<float[][]> inputdata = new ArrayList<>(hiddenlayer_N+1);
		inputdata.add(x);

		FloatMatrix z= x.dup(); //各層の出力値を格納、ロジスティック層の値を格納
//		float[][] z= new float[0][0], dy; //各層の出力値を格納、ロジスティック層の値を格納

		//順伝播
		for(int layer=0; layer<hiddenlayer_N; layer++){
			//float[] data; //入力値を保持
			//float[][] z_ = new float[minibatchSize][hidden_N[layer]];
			FloatMatrix data = z.dup();

			z = sigmoidlayer[layer].forward(data);

			/*for(int n=0; n<minibatchSize; n++){
				if(layer == 0){
					//入力層ではdataにx[0]の値を代入
					data = x[n];
				}else{
					data = z[n];
				}
				z_[n] = sigmoidlayer[layer].forward(data);
			}
			z = z_;*/
			inputdata.add(z.dup());
		}

		//出力層にて、順伝播、逆伝播を行う。
		FloatMatrix dy = logisticlayer.train(z, label, minibatchSize, l_rate);
		FloatMatrix dz = new FloatMatrix();
		//逆伝播用変数
		FloatMatrix weight_prev;
//		float[][] weight_prev;
//		float[][] dz= new float[0][0];

		for(int layer = hiddenlayer_N -1; layer >= 0; layer--){

			if(layer == hiddenlayer_N-1){
				//逆伝播の一番最初(出力層)の場合
				weight_prev = logisticlayer.weight.dup();
			}else{
				weight_prev = sigmoidlayer[layer+1].weight.dup();
				dy = dz.dup();
			}
			dz = sigmoidlayer[layer].backward(inputdata.get(layer), inputdata.get(layer+1), dy, weight_prev, minibatchSize, l_rate);
			dae[layer].weight = sigmoidlayer[layer].weight.dup();
		}

	}

	/**
	 * プレトレーニングメソッド
	 * @param x ミニバッチに分けたトレーニングデータ
	 * @param minibatchSize ミニバッチサイズ
	 * @param minibatch_N ミニバッチ数
	 * @param epochs 学習回数
	 * @param l_rate 学習率
	 * @param noise_lvl 入力データにノイズを加える確率
	 */
	public void pretrain(FloatMatrix[] x, int minibatchSize, int minibatch_N, int epochs, float l_rate, float noise_lvl) {
//		public void pretrain(float[][][] x, int minibatchSize, int minibatch_N, int epochs, float l_rate, float noise_lvl) {
		// レイヤーワイズ（層ごとの学習）によるトレーニング
		for(int layer=0; layer<hiddenlayer_N; layer++){
			for(int epoch=0; epoch<epochs; epoch++){
				for(int batch=0; batch< minibatch_N; batch++){
					FloatMatrix data = new FloatMatrix(minibatchSize,input_N);
					FloatMatrix prelayer_data;
//					float[][] data = new float[minibatchSize][input_N];
//					float[][] prelayer_data;

					if(layer==0){
						data = x[batch].dup();
					}else{
						prelayer_data = data.dup();
//						data = new float[minibatchSize][hidden_N[layer-1]];
						data = sigmoidlayer[layer-1].output(prelayer_data);
//						for(int i=0; i<minibatchSize; i++){
//							data[i] = sigmoidlayer[layer-1].output(prelayer_data[i]);
//						}
					}
					//プレトレーニング実行
					dae[layer].train(data, minibatchSize, l_rate, noise_lvl);
					sigmoidlayer[layer].weight = dae[layer].weight.dup();
				}
			}
		}

//		for(int layer=0; layer<hiddenlayer_N; layer++){
//			System.out.println("s["+layer+"]:"+sigmoidlayer[layer].weight);
//			System.out.println("a["+layer+"]:"+dae[layer].weight.dup());
//		}


	}

	/**
	 * クラスを予測する
	 * @param x 入力データ
	 * @return 予測結果
	 */
	public FloatMatrix predict(FloatMatrix x){
//		public Integer[] predict(float[] x){
		FloatMatrix z= x.dup();
//		float[] z= new float[0];

		for(int layer=0; layer < hiddenlayer_N; layer++){
			FloatMatrix x_ = z.dup(); //各層の出力用
//			float[] x_; //各層の出力用

//			if(layer == 0){
//				x_ = x;
//			}else{
//				x_ = z.clone();
//			}
			z = sigmoidlayer[layer].forward(x_);
		}
		return logisticlayer.predict(z);
	}

}
