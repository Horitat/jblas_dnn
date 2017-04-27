package test_ujmp;

import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class Sample_ujmp {
/*
 * Matrix.Factory.eye(row,clumn)で単位行列
 * x.plus(Ret.ORIG, true, Matrix or scalar)でxに計算結果が入る
 * MathUtilにガウス分布あり
 *getValueCount()で総要素数,getZCount()で三次元目の要素数
 *getSize(int n)でn次元目の要素数
 */
	public static void main(String[] args) {
		// TODO 自動生成されたメソッド・スタブ
		Matrix x = DenseMatrix.Factory.linkToArray(
				new double[] {1, 2},
				new double[] {3, 4}
				);

		Matrix y = DenseMatrix.Factory.linkToArray(
				new double[] {5, 6},
				new double[] {7, 8}
				);
		// (a)要素ごとの足し算
		System.out.println( x.plus(y) );
		System.out.println("-----");
		// (a)スカラーの足し算
		Matrix t = x.plus(8);
		System.out.println( t );
		System.out.println("-----");
		// (b)行列の掛け算
		System.out.println( x.mtimes(y) );

		System.out.println("-----");
		// (c)転置行列
		System.out.println( x.transpose() );

		System.out.println("-----");
		// (c)要素ごとの掛け算
		System.out.println( x.times(y) );

		System.out.println("-----");
		// (c)要素ごとの割り算
		System.out.println( y.divide(x) );
		System.out.println("-----");
		// (c)要素ごとの割り算
		System.out.println( y.divide(4) );



		System.out.println("-----");
		// (c)要素ごとの割り算
//		Matrix hh = y.transpose();
//		System.out.println( hh );

		System.out.println("-----");
		// (c)要素ごとの割り算

		Matrix rand = Matrix.Factory.rand(10,10);
		//System.out.println( rand );
		long n = 2;
//		rand.showGUI();

		System.out.println("-----");

		Matrix pp = Matrix.Factory.rand(n, n+1, n+2, n);
		//Matrix fa = Matrix.Factory.rand(n, n+1, n+2, 2);
		pp.plus(Ret.ORIG,true,6.f);
		pp.times(Ret.ORIG,true,6.f);
		System.out.println(pp.getDimensionCount());
		System.out.println(pp.transpose(Ret.NEW).getDimensionCount());
		System.out.println(pp.getAsFloat(0,0,0,0));

//		FloatMatrix z = (FloatMatrix) Matrix.Factory.sparse(ValueType.FLOAT, n, n+1, n+2, n);
//		z.setAsFloat(4.f, 0,0,0,0);
//		//z.setColumnLabel(1, 8);
		System.out.println("-----");

		Matrix test = pp.reshape(Ret.NEW, (n*(n+1)), (n+2)*n);
//		Matrix test = pp.reshape(Ret.NEW, 48, 1);

		System.out.println(test.getDimensionCount());
		System.out.println( test.getRowCount()+"::::"+ test.getColumnCount());
//		System.out.println(test);
		System.out.println("-----");

		//z.showGUI();
		//double qr = y.det();
//		Matrix l = Matrix.Factory.zeros(n,n+1,n+2,n).transpose(Ret.NEW);
		Matrix l = test.reshape(Ret.NEW, n,n+1,n+2,n);
		System.out.println(l.getDimensionCount());
		System.out.println( l.getAsDouble(0,1));
		System.out.println("****************");
		//l = Matrix.Factory.copyFromMatrix(pp);//.transpose(Ret.NEW);//.times(Ret.NEW,true,1);
		Matrix cc = pp.times(Ret.NEW, false, 1);
		System.out.println(pp.times(Ret.ORIG, false, 1.f).getDimensionCount());
		//System.out.println( l.getAsDouble(0,0,0,0));
		System.out.println(cc.getDimensionCount());
		System.out.println( cc);
		System.out.println( pp.getRowCount()+"::::"+ pp.getColumnCount());
		System.out.println(pp.isDiagonal());

		System.out.println(cc.getAsMatrix(1,2));
//		BufferedImage[] img = Read_img.read_img_buffer("C:\\pleiades\\originaldnn_test_data\\",227,227,false);
//		Matrix[] imageMatrix = new ImageMatrix[img.length];
//
//		for(int i=0; i< img.length; i++){
//			imageMatrix[i] = new ImageMatrix(img[i]);
//		}


		//ujmpなら可能。これで4次元テンソルが描ける
		//実際に計算に利用するのは下二次元の重みのみ
		Matrix[][] kkk = new Matrix[4][4];
		for(int i = 0; i < kkk.length; i++){
			for(int k=0; k<kkk[0].length; k++){
				kkk[i][k] = Matrix.Factory.rand(2,3);
			}
		}

		for(int i = 0; i < kkk.length; i++){
			for(int k=0; k<kkk[0].length; k++){
	//			System.out.println(kkk[i][k]);
				System.out.println("****************");
			}
		}





		// create a very large sparse matrix
//		SparseMatrix m1 = SparseMatrix.Factory.zeros(1000000, 500000);

		// set some values
//		m1.setAsDouble(MathUtil.nextGaussian(), 0, 0);
//		m1.setAsDouble(MathUtil.nextGaussian(), 1, 1);
//		for (int i = 0; i < 10000; i++) {
//			m1.setAsDouble(MathUtil.nextGaussian(), MathUtil.nextInteger(0, 1000), MathUtil.nextInteger(0, 1000));
//		}
		//System.out.println( m1);


//		m1.showGUI();
	}

}
