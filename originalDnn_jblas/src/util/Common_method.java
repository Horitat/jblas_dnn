package util;

import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

import org.jblas.FloatMatrix;

import Mersenne.Sfmt;

public class Common_method {

	/**
	 * 配列をシャッフルする
	 * */
	public static <T> void array_shuffle(T[] array, Sfmt mt){
		for (int i = 0; i < array.length; i++) {
			int dst = mt.NextInt(array.length);
			T tmp = array[i];
			array[i] = array[dst];
			array[dst] = tmp;
		}
	}


	/**
	 * リストをシャッフルする
	 * */
	public static void list_shuffle(List list, Sfmt mt){
		Object[] array = list.toArray();
		for (int i = 0; i < array.length; i++) {
			int dst = mt.NextInt(array.length);
			Object tmp = array[i];
			array[i] = array[dst];
			array[dst] = tmp;
		}
		ListIterator it = list.listIterator();
		list.clear();
		for (int i=0; i<array.length; i++) {
//			it.next();
//			it.set(array[i]);
			//System.out.println("--------------------------------------------");
			list.add(array[i]);
		}

	}

/**
 * モデルの結果を出力
 * @param predicted_T 予測結果
 * @param test_T 正解ラベル
 */
	public static void print_result(FloatMatrix predicted_T, FloatMatrix test_T){
//		FloatMatrix predicted_T = classifier.predict(test_X);
		double accuracy = 0.;
		int test_N = predicted_T.rows, patterns = predicted_T.columns;
		FloatMatrix confusionMatrix = new FloatMatrix(patterns,patterns);
		FloatMatrix precision = new FloatMatrix(patterns);
		FloatMatrix recall = new FloatMatrix(patterns);

		for (int i = 0; i < test_N; i++) {
			Integer[] predict_array = new Integer[predicted_T.getRow(i).toIntArray().length];
			Integer[] label_array = new Integer[test_T.getRow(i).toIntArray().length];

			for(int j=0; j<predict_array.length; j++){
				predict_array[j] = predicted_T.getRow(i).toIntArray()[j];
				label_array[j] =test_T.getRow(i).toIntArray()[j];
			}

			int predicted_ = Arrays.asList(predict_array).indexOf(1);
			int actual_ = Arrays.asList(label_array).indexOf(1);

			if(predicted_ != -1 && actual_ != -1){
//				confusionMatrix[actual_][predicted_] += 1;
			}
			confusionMatrix.put(predicted_,actual_, confusionMatrix.get(predicted_,actual_)+1f);
		}

		for(int i=0; i<patterns; i++){
			accuracy += confusionMatrix.get(i,i);
			precision.put(i, (precision.get(i)+confusionMatrix.get(i,i))/confusionMatrix.getColumn(i).sum());
			recall.put(i, (recall.get(i)+confusionMatrix.get(i,i))/confusionMatrix.getRow(i).sum());
		}

		accuracy /= test_N;

		System.out.println("------------------------");
		System.out.println("model evaluation");
		System.out.println("------------------------");
		System.out.printf("Accuracy: %.1f %%\n", accuracy * 100);
		System.out.println("Precision:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, precision.get(i) * 100);
		}
		System.out.println("Recall:");
		for (int i = 0; i < patterns; i++) {
			System.out.printf(" class %d: %.1f %%\n", i+1, recall.get(i) * 100);
		}
	}



	/**
	 *畳み込み層とプーリング層の出力サイズを計算する。
	 * @param input_N 入力数
	 * @param kernel カーネルサイズ
	 * @param stride スライド数
	 * @param padding ゼロパディングの数
	 * @return
	 */
	public static int compute_outputN(int input_N, int kernel, int stride, int padding){
		return ((input_N-kernel+2*padding)/stride)+1;
	}


	@FunctionalInterface
	public interface FloatFunction<R> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(float value);
	}

	@FunctionalInterface
	public interface FloatFunction2<R, S,T, U> {

		/**
		 * Applies this function to the given argument.
		 *
		 * @param value the function argument
		 * @return the function result
		 */
		R apply(float dw, float w, float learning_rate, float momentum);
	}
}
