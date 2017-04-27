package read_inputdata;

import java.awt.Graphics2D;
import java.awt.Toolkit;
import java.awt.image.AreaAveragingScaleFilter;
import java.awt.image.BufferedImage;
import java.awt.image.FilteredImageSource;
import java.awt.image.ImageFilter;
import java.awt.image.ImageProducer;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

import javax.imageio.ImageIO;

public class Read_img {

	/*
	 *後々、4次元配列を返すように扱う
	 *[トレーニングかテストのデータ数][チャンネル][サイズ][サイズ]
	 *その時に、get_files_tostringやディレクトリチェックを入れる
	 *
	 */

	/**
	 * カラー画像を読み取り配列に入れる。チャネルは3固定
	 * @param input 読み取りたい画像が入っているフォルダパス
	 * @return 読み取った画像
	 */
	public static int[][][][] read_color_img(String input, int wid, int hei){
		System.out.println("Read image data");
		BufferedImage[] inputimg;
		try {
			File check = new File(input);

			if(!check.isDirectory()){
				System.out.println("Specify path["+input+"] is not directory");
				System.exit(1);
			}

			String[] filename = get_files_tostring(input);
			inputimg = new BufferedImage[filename.length];
			System.out.println("read in "+ input + " "+ filename.length +" files");
			//入力画像の縦横のサイズに0が入力されていれば入力画像の縦横の最大値をセットする
			if(wid <= 0 || hei <= 0){
				if(wid <= 0 && hei <= 0){
					wid = get_imgsize(inputimg, input, filename);
					hei = wid;
				}else if(wid <= 0){
					wid = hei;
				}else if(hei <= 0){
					hei = wid;
				}
			}

			int[][][][] picture = new int[filename.length][3][wid][hei];
			//System.out.println(pixel.length);
			for(int k=0; k<filename.length; k++){
				//サイズに0が入力されていなければ画像を読み込み
				if(inputimg[k] == null){
					inputimg[k] = ImageIO.read(new File(input+ filename[k]));
				}
				//System.out.println(inputimg.getWidth()+":"+inputimg.getHeight());
				//入力画像と設定したサイズが違えばリサイズ
				if((wid != inputimg[k].getWidth() || hei != inputimg[k].getHeight())){
					//System.out.println(wid+"::::"+hei);
					inputimg[k] = resize_img(new File(input+filename[k]), wid, hei);
				}
				int[] pixel = inputimg[k].getRGB(0,0,wid,hei,null,0,wid);
				for(int i=0; i<wid; i++){
					int n = hei * i;
					for(int j=0; j<hei; j++){
						int argb = pixel[n+j];

						picture[k][0][i][j] = argb >> 16 &0xFF;
						picture[k][1][i][j] = argb >> 8 &0xFF;
						picture[k][2][i][j] = argb >> 0 &0xFF;
				//System.out.println((n+j)+",");
					}
				}
			}
			//System.out.println(picture[0][0].length);

			return picture;
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		return null;
	}


	/**
	 * カラー画像を読み取り配列に入れる。チャネルは3固定
	 * @param input 読み取りたい画像が入っているフォルダパス
	 * @param wid 入力する画像サイズ。元画像とサイズが違うとリサイズ
	 * @param hei 入力する画像サイズ。元画像とサイズが違うとリサイズ
	 * @return 読み取った画像
	 */
	public static float[][][][] read_color_img_to_gray(String input, int wid, int hei){
		BufferedImage[] inputimg;
		try {
			File check = new File(input);
			if(!check.isDirectory()){
				System.out.println("Specify path["+input+"] is not directory");
				System.exit(1);
			}

			String[] filename = get_files_tostring(input);
			inputimg = new BufferedImage[filename.length];
			System.out.println("read in "+ input + " "+ filename.length +" files");
			//入力画像の縦横のサイズに0が入力されていれば入力画像の縦横の最大値をセットする
			if(wid <= 0 || hei <= 0){
				if(wid <= 0 && hei <= 0){
					wid = get_imgsize(inputimg, input, filename);
					hei = wid;
				}else if(wid <= 0){
					wid = hei;
				}else if(hei <= 0){
					hei = wid;
				}
			}

			float[][][][] picture = new float[filename.length][1][wid][hei];
			for(int k=0; k<filename.length; k++ ){
				//System.out.println(filename[k]);
				//サイズに0が入力されていなければ画像を読み込み
				if(inputimg[k] == null){
					inputimg[k] = ImageIO.read(new File(input+ filename[k]));
				}
				//System.out.println(inputimg.getWidth()+":"+inputimg.getHeight());
				//入力画像と設定したサイズが違えばリサイズ
				if((wid != inputimg[k].getWidth() || hei != inputimg[k].getHeight())){
					//System.out.println(wid+"::::"+hei);
					inputimg[k] = resize_img(new File(input+filename[k]), wid, hei);
				}
				//System.out.println(inputimg.getWidth()+":"+inputimg.getHeight());

				//System.out.println(pixel.length);
				for(int i=0; i<wid; i++){
					for(int j=0; j<hei; j++){
						int argb = inputimg[k].getRGB(i,j);
						//グレースケールに変換
						picture[k][0][i][j] = (float)(0.299*(argb >> 16 &0xff) + 0.587 * (argb >> 8 & 0xff) + 0.114 * (argb & 0xff) + 0.5);
						//System.out.println((n+j)+",");
						picture[k][0][i][j] = ( ((int)picture[k][0][i][j] <<16) | ((int)picture[k][0][i][j] <<8) | ((int)picture[k][0][i][j]) );
					}
				}
				//				System.out.println(wid+":"+hei);
				//System.out.println(picture[0][0].length+":"+picture[0][0][0].length);
			}

			return picture;
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		System.out.println(wid+"::::"+hei);
		return null;
	}



	/**
	 * カラー画像を読み取りBufferedImageに入れる。
	 * @param input 読み取りたい画像が入っているフォルダパス
	 * @param wid 入力する画像サイズ。元画像とサイズが違うとリサイズ
	 * @param hei 入力する画像サイズ。元画像とサイズが違うとリサイズ
	 * @param color TrueならRGB、Falseならグレースケール
	 * @return 読み取った画像
	 */
	public static BufferedImage[] read_img_buffer(String input, int wid, int hei, boolean color){
		BufferedImage[] inputimg, gray_img;
		try {
			File check = new File(input);
			if(!check.isDirectory()){
				System.out.println("Specify path["+input+"] is not directory");
				System.exit(1);
			}

			String[] filename = get_files_tostring(input);
			inputimg = new BufferedImage[filename.length];
			gray_img = new BufferedImage[filename.length];


			System.out.println("read in "+ input + " "+ filename.length +" files");
			//入力画像の縦横のサイズに0が入力されていれば入力画像の縦横の最大値をセットする
			if(wid <= 0 || hei <= 0){
				if(wid <= 0 && hei <= 0){
					wid = get_imgsize(inputimg, input, filename);
					hei = wid;
				}else if(wid <= 0){
					wid = hei;
				}else if(hei <= 0){
					hei = wid;
				}
			}

			int[][][] picture;

			if(color){
				picture = new int[3][wid][hei];
			}else{
				picture = new int[1][wid][hei];
			}
			for(int k=0; k<filename.length; k++ ){
				//サイズに0が入力されていなければ画像を読み込み
				if(inputimg[k] == null){
					inputimg[k] = ImageIO.read(new File(input+ filename[k]));

				}
				//入力画像と設定したサイズが違えばリサイズ
				if((wid != inputimg[k].getWidth() || hei != inputimg[k].getHeight())){
					inputimg[k] = null;
					inputimg[k] = resize_img(new File(input+filename[k]), wid, hei);
//					System.out.println(inputimg[k].getRGB(100,100)+":sss:"+inputimg[k].getRGB(100,100)+":"+k+":"+inputimg[k].getType());
				}

				if(!color){
					gray_img[k] = new BufferedImage(wid,hei,BufferedImage.TYPE_BYTE_GRAY);
					for(int i=0; i<wid; i++){
						for(int j=0; j<hei; j++){
							int argb = inputimg[k].getRGB(i,j);
							//グレースケールに変換

							picture[0][i][j] = (int)(0.299*(argb >> 16 &0xff) + 0.587 * (argb >> 8 & 0xff) + 0.114 * (argb & 0xff) + 0.5);
							picture[0][i][j] = (((picture[0][i][j] <<16))
									| ((picture[0][i][j] <<8)) | ((picture[0][i][j])));
							gray_img[k].setRGB(i, j, picture[0][i][j]);

						}
					}

					//確認用
					ImageFilter filter = new AreaAveragingScaleFilter(wid, hei);
					ImageProducer p = new FilteredImageSource(gray_img[k].getSource(), filter);
					java.awt.Image dstImage = Toolkit.getDefaultToolkit().createImage(p);

					Graphics2D g = gray_img[k].createGraphics();
					g.drawImage(dstImage, 0, 0, null);
					g.dispose();

					//ImageIO.write(gray_img[k], "jpg", new File("C:\\pleiades\\originaldnn_test_data\\grayimag"+k+".jpg"));
					//ここまで
				}
			}

			if(color){
				return inputimg;
			}else{
				return gray_img;
			}
		} catch (IOException e) {
			// TODO 自動生成された catch ブロック
			e.printStackTrace();
		}
		System.out.println(wid+"::::"+hei);
		return null;
	}



	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
//		String input_folderpath = "C:\\pleiades\\originaldnn_test_data\\mnistdata\\";
//		String input_testfilepath = "C:\\pleiades\\originaldnn_test_data\\mnistdata\\test\\"
//				, input_trainfilepath = "C:\\pleiades\\originaldnn_test_data\\mnistdata\\train\\";
//
//		int train_N = Read_number_of_data.count_data(input_folderpath + "mnist_train.txt");
//		int test_N = Read_number_of_data.count_data(input_folderpath + "mnist_test.txt");
//		//テスト用
//		//read_color_img("C:\\pleiades\\workspace\\conv_output.jpg");
//		//read_color_img_to_gray("C:\\pleiades\\workspace\\conv_output.jpg", 32, 32);
//		//read_color_img("C:\\pleiades\\workspace\\conv_output.jpg");
//		float[][][][] input_traindata = read_color_img_to_gray(input_trainfilepath, 0, 0);
//		float[][][][] input_testdata = read_color_img_to_gray(input_testfilepath, 0, 0);
//		//************
//
//		if(input_traindata.length != train_N){
//			System.out.println("number of Train data error. read img file is "+ input_traindata.length +
//					". read txt file is "+ train_N);
//			System.exit(1);
//		}
//
//		if(input_testdata.length != test_N){
//			System.out.println("number of Test data error. read img file is "+ input_testdata.length +
//					". read txt file is "+ test_N);
//			System.exit(1);
//		}
//
//		System.out.println(input_traindata.length+":"+input_traindata[0].length+ ":"+input_traindata[0][0].length+":"+input_traindata[0][0][0].length);
//		System.out.println(input_testdata.length+":"+input_testdata[0].length+ ":"+input_testdata[0][0].length+":"+input_testdata[0][0][0].length);


		String folder = "C:\\pleiades\\originaldnn_test_data\\";
		String file = folder+ "testpic.txt";
		read_img_buffer(folder, 256,256, false);


		/*
		switch(args.length){
		case 0:
			input_folderpath = "C:\\Users\\WinGAIA\\Desktop\\割れ写真判定学習用512\\まとめ_bk\\傷汚れなし\\";
			input_testfilepath = input_folderpath + "test_data\\";
			input_trainfilepath = input_folderpath + "train_data\\";
			break;

		case 1:
			input_folderpath = args[1];
			input_testfilepath = input_folderpath + "test_data\\";
			input_trainfilepath = input_folderpath + "train_data\\";
			break;

		case 3:
			input_folderpath = args[1];
			input_testfilepath = input_folderpath + args[2];
			input_trainfilepath = input_folderpath + args[3];

		default:
			System.err.println("引数でフォルダを指定してください");
		}


		int[][][] traindata = read_color_img(input_trainfilepath);
		int[][][] testdata = read_color_img(input_testfilepath);
		 */
		System.out.println("end create input data");
	}

	/**
	 * フォルダにあるjpgファイルを獲得する
	 * @param folderpath フォルダパス
	 * @return ファイル一覧
	 */
	public static String[] get_files_tostring(String folderpath){
		File files = new File(folderpath);

		if(!files.isDirectory()){
			System.out.println("getting is File!!" + folderpath);
		}
		//ファイル一覧を取得、取得はファイル名＋拡張子のみ
		return files.list(new Filter_png_and_jpg());
	}


	static int pop = 0;
	private static BufferedImage resize_img(File input, int width,int height) throws IOException{

		if(!input.isFile()){
			System.out.println("input is not file");
			return null;
		}

		BufferedImage org = ImageIO.read(input);
		ImageFilter filter = new AreaAveragingScaleFilter(width, height);
		ImageProducer p = new FilteredImageSource(org.getSource(), filter);
		java.awt.Image dstImage = Toolkit.getDefaultToolkit().createImage(p);

		BufferedImage dst = new BufferedImage(dstImage.getWidth(null), dstImage.getHeight(null),
				BufferedImage.TYPE_INT_RGB);
		dst.getGraphics().drawImage(dstImage, 0, 0, null);
		dst.getGraphics().dispose();
		System.out.println(dst.getRGB(100,100)+"::read::"+org.getRGB(100, 100));
//		ImageIO.write(dst, "jpg", new File("C:\\pleiades\\originaldnn_test_data\\test"+pop+".jpg"));
//		pop++;
		return dst;
	}


	private static int get_imgsize(BufferedImage[] img, String input, String[] name) throws IOException{

		//BufferedImage inputimg;
		int size = 0;
		for(int i=0; i<name.length; i++){
			img[i] = ImageIO.read(new File(input+ name[i]));
			System.out.println(img[i].getHeight());
			if(size < img[i].getHeight()){
				size = img[i].getHeight();
			}

			if(size < img[i].getWidth()){
				size = img[i].getWidth();
			}
		}

		return size;
	}

}

class Filter_jpg implements FilenameFilter{
	public boolean accept(File dir, String name){

		if(name.matches(".*jpg$") || name.matches(".*JPG$")){
			return true;
		}
		//System.out.println(name);
		return false;
	}
}

class Filter_png implements FilenameFilter{
	public boolean accept(File dir, String name){

		if(name.matches(".*png$") || name.matches(".*PNG$")){
			return true;
		}
		//System.out.println(name);
		return false;
	}
}

class Filter_png_and_jpg implements FilenameFilter{
	public boolean accept(File dir, String name){

		if(name.matches(".*png$") || name.matches(".*PNG$") || name.matches(".*jpg$") || name.matches(".*JPG$")){
			return true;
		}
		//System.out.println(name);
		return false;
	}
}
