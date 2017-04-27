package read_inputdata;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import org.apache.commons.lang3.math.NumberUtils;

public class Read_label {

	/**
	 * 分類の場合の教師値を取る。
	 * 注意としては、読み込ませた画像の順番で教師値が並んでいなければならない
	 * ファイルの書き方は[dataname label]
	 * @param path 教師値が書かれているファイル
	 * @param num 訓練データの個数
	 * @return 訓練データ分の教師値
	 */
	public static int[][] Classifier_label(String path, int num){
		System.out.println("Read classifier label");
		BufferedReader br1 = null;

		int[] data = new int[num];

		try {
			File file1 = new File(path);

			if(!file1.isFile()){
				System.out.println("Specify path["+path+"] is not file");
			}

			br1 = new BufferedReader(new FileReader(file1));

			String str = br1.readLine();
			int max=1, n=0;
			while(str != null ){
				String l = str.split(" ")[1];

				if(!NumberUtils.isDigits(l)){
					System.out.println("Error:"+ str);
					System.out.println("Label must be Number");
					System.exit(1);
				}

				if(max < Integer.parseInt(l)+1){
					max = Integer.parseInt(l)+1;
				}
				data[n] = Integer.parseInt(l);
				n++;
				str = br1.readLine();
			}


			int[][] label = new int[num][max];

			n=0;
			for(int i :data){
				label[n][i] = 1;
				n++;
			}

			br1.close();
			return label;
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		return null;
		}

	/**
	 * 回帰問題の場合、複数のラベルも可能
	 * ファイルの書き方は[dataname label1 label2 ....]
	 * @param path ラベルが書かれているファイルのパス
	 * @param num データ数
	 * @return データ数に対するラベル
	 */
	public static float[][] Regression_label(String path, int num){
		System.out.println("Read regression label");
		BufferedReader br1 = null;

		float[][] label = null;

		try {
			File file1 = new File(path);

			if(!file1.isFile()){
				System.out.println("Specify path["+path+"] is not file");
			}

			br1 = new BufferedReader(new FileReader(file1));

			String str = br1.readLine();
			int n=0;
			String[] data = str.split(" ");

			//一番最初の文字列をデータ名と仮定して、-1
			int len = data.length-1;
			label = new float[num][len];

			while(str != null ){

				data = str.split(" ");
				for(int i=0; i<len; i++){
					String l = data[i+1];
					if(!NumberUtils.isNumber(l)){
						System.out.println("Error:"+ str);
						System.out.println("Label must be Number:"+l);
						System.exit(1);
					}
					label[n][i] = Float.parseFloat(l);
				}

				n++;
				str = br1.readLine();

			}

			br1.close();
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		return label;
	}


	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		String class_path = "C:\\pleiades\\originaldnn_test_data\\mnist_test.txt";
		String reg_path = "C:\\pleiades\\originaldnn_test_data\\regtest.txt";

		int train_class_N = Read_number_of_data.count_data(class_path);
		int train_reg_N = Read_number_of_data.count_data(reg_path);

		int[][] classifier = Classifier_label(class_path, train_class_N);
		float[][] reg = Regression_label(reg_path, train_reg_N);

		System.out.println("classifier test");
		for(int i=0; i<classifier.length; i++){
			for(int j=0; j<classifier[0].length; j++)
				System.out.println("["+i+"]["+j+"]"+classifier[i][j]);
		}

		System.out.println("regression test");
		for(int i=0; i<reg.length; i++){
			for(int j=0; j<reg[0].length; j++){
				System.out.print(reg[i][j]+",");
			}
			System.out.println();
		}
		//Classifier or Regression check
	}

}
