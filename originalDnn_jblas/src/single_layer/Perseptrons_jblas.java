package single_layer;

import org.jblas.FloatMatrix;

import util.ActivationFunction;
import util.GaussianDistribution;
import Mersenne.Sfmt;

public class Perseptrons_jblas {
//	private float[] weight;
	private FloatMatrix weight;
	public int input_N;


	public Perseptrons_jblas(int input){
		input_N = input;
//		weight = new float[input];
		weight = new FloatMatrix( new float[input]);
	}

	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		int[] init_key = {(int) System.currentTimeMillis(), (int) Runtime.getRuntime().freeMemory()};
		//Sfmt mt = new Sfmt((int)(Runtime.getRuntime().freeMemory()/(1024)));
		Sfmt mt = new Sfmt(init_key);


		final int train_N = 1000;//学習データの数
		final int test_N = 200;//テストデータの数
		int input_N = 2;//入力の数

		FloatMatrix train_data = new FloatMatrix( new float[train_N][input_N]);
		FloatMatrix train_label = new FloatMatrix( new float[train_N]);//学習データのラベル

		FloatMatrix test_data = new FloatMatrix(new float[test_N][input_N]);//テストデータ
		FloatMatrix test_label = new FloatMatrix(new float[test_N]);//テストデータのラベル
		FloatMatrix predict = new FloatMatrix(new float[test_N]);//予測結果

		final int epochs = 2000;//トレーニングの最大世代数
		final float l_rate = 1.0f;//学習率


		GaussianDistribution g1 = new GaussianDistribution(-2., 1., mt);
		GaussianDistribution g2 = new GaussianDistribution(2., 1., mt);

		for(int i=0; i<train_N; i++){
			for(int j=0; j<input_N; j++){
				//train_data[i][j] = (float)g1.random();
				if(i < train_N/2){
					if(j % 2 == 0){
						train_data.put(i,j,(float)g1.random());
					}else{
						train_data.put(i,j,(float)g2.random());
					}
					train_label.put(i,1.f);
				}else{
					if(j % 2 == 0){
						train_data.put(i,j,(float)g2.random());
					}else{
						train_data.put(i,j,(float)g1.random());
					}
					train_label.put(i,-1);

				}
			}
		}

		for(int i=0; i<test_N; i++){
			for(int j=0; j<input_N; j++){
				//train_data[i][j] = (float)g1.random();
				if(i < test_N/2){
					if(j % 2 == 0){
						test_data.put(i,j,(float)g1.random());
					}else{
						test_data.put(i,j,(float)g2.random());
					}
					test_label.put(i,1);
				}else{
					if(j % 2 == 0){
						test_data.put(i,j,(float)g2.random());
					}else{
						test_data.put(i,j,(float)g1.random());
					}
					test_label.put(i,-1);
				}
			}
		}
		int epoch = 0;
		int test_interval = epochs / 5;
		Perseptrons_jblas classifier = new Perseptrons_jblas(2);
		classifier.weight.put(0, (float)mt.NextNormal());
		classifier.weight.put(1, (float)mt.NextNormal());
		//学習のループ
		while(epoch < epochs){
			int end_flg = 0;

			for(int i=0; i<train_N; i++){
				end_flg += classifier.train(train_data.getRow(i), train_label.getRow(i), l_rate);
			}

			if(end_flg == train_N){
				break;
			}

			if(epoch % test_interval == 0){

			}

			epoch++;
		}
		//テスト実行
		for(int i=0; i<test_N; i++){
			predict.put(i,classifier.predict(test_data.getRow(i)));
		}

		FloatMatrix confusion = new FloatMatrix(new float[input_N][input_N]);
		//正答率、精度、再現率
		float accuracy=0f, precision=0f, recall=0f, conf = 0;
		//テストの結果を評価
		for(int i=0; i<test_N; i++){
			if(predict.get(i) > 0){
				if(test_label.get(i) > 0){
					accuracy += 1;
					precision += 1;
					recall += 1;
					conf = confusion.get(0, 0) + 1.f;
					confusion.put(0, 0, conf);
				}else{
					confusion.put(1,0,confusion.get(1,0) + 1);
				}
			}else{
				if(test_label.get(i) > 0){
					confusion.put(0, 1, confusion.get(0,1)+1);
				}else{
					accuracy += 1;
					confusion.put(1,1,confusion.get(1,1) + 1);
				}
			}
		}

		accuracy = accuracy / test_N;
		precision = precision / (confusion.get(0,0) + confusion.get(1,0));
		recall = recall /(confusion.get(0,0) + confusion.get(0,1));

		System.out.println("----------------------------");
		System.out.println("Perceptrons model evaluation");
		System.out.println("----------------------------");
		System.out.printf("Accuracy:  %.1f %%\n", accuracy * 100);
		System.out.printf("Precision: %.1f %%\n", precision * 100);
		System.out.printf("Recall:    %.1f %%\n", recall * 100);
	}

	/**
	 * 入力を、重みづけで加算し活性化関数の結果を返す
	 * */
	public float predict(FloatMatrix data) {
//		public int predict(float[] data) {
		// TODO 自動生成されたメソッド・スタブ

		float activation = data.mmul(weight).get(0);

//		for(int i=0; i< input_N; i++){
//			activation += weight[i] * data[i];
//		}
		//活性化関数
		return ActivationFunction.step_function(activation);
	}

	public int train(FloatMatrix data, FloatMatrix label, float l_rate) {
		// TODO 自動生成されたメソッド・スタブ
//		public int train(float[] data, int label, float l_rate) {
		int result = 0;
		float c = data.mmul(weight).get(0) * label.get(0);
//		System.out.println("----------data------------------");
//		System.out.println(data);
//		System.out.println(data.transpose());
//		System.out.println("----------weight------------------");
//		System.out.println(weight);
//		System.out.println("----------mmul------------------");
//		System.out.println(data.mmul(weight));
//		System.out.println("----------mul------------------");
//		System.out.println(weight.mmul(data));
//		System.out.println(label);
//		if(data.mmul(weight) == weight.mmul(data)){
//			System.out.println("true");
//		}
		//出力の結果を確認
		//for(int i=0; i<input_N; i++){
//			c += weight[i] * data[i] * label;
		//}
//		System.out.println(data.mul(label));
//		System.out.println("----------tran------------------");
//		System.out.println(data.transpose().mul(label));
		//出力が正しくなければ重みを更新
		if(c > 0){
			result = 1;
		}else{
//			for(int i=0; i<input_N; i++){
//				weight[i] += l_rate * data[i] * label;
//			}
			weight.addi(data.transpose().mul(label).mul(l_rate));
		}


		return result;
	}

}
