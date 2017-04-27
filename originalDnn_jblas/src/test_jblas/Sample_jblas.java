package test_jblas;

import org.jblas.DoubleMatrix;

public class Sample_jblas {

	public static void main(String[] args) {
		DoubleMatrix x = new DoubleMatrix(new double[][] {
				{1, 2},
				{3, 4}
		});

		DoubleMatrix y = new DoubleMatrix(new double[][] {
				{5, 6},
				{7, 8}
		});

		DoubleMatrix v = new DoubleMatrix(new double[][] {
				{5, 6,1,2},
				{7, 8,3,4},
				{9,10,13,14},
				//{11,12,15,16}
//				{20,21,22,24}
		});

		// (a)
		System.out.println("x+y");
		System.out.println( x.add(y) );

		System.out.println("xy");
		// (b)行列の掛け算
		System.out.println( x.mmul(y) );
		System.out.println("x*y");
		// (b)要素ごと
		System.out.println( x.mul(y) );
		System.out.println("x^t");
		// (c)
		System.out.println( x.transpose() );
		System.out.println("y(1,4)=14");
		y.put(3, 14);
		System.out.println( y );
		System.out.println("v-1");
		System.out.println( v.sub(1) );
		System.out.println("v");
		System.out.println( v.rows+"****"+ v.columns );
		System.out.println("func");
		//DoubleMatrix f = test(x,y);
		System.out.println( x );
//		System.out.println( x.subColumnVector(DoubleMatrix.ones(2)) );
		System.out.println("func");
		//y= y.add(5);
		System.out.println( y );
		System.out.println("end");

		DoubleMatrix[][] k = new DoubleMatrix[4][2];

		for(int i=0; i<k.length; i++){
			for(int j=0; j<k[0].length; j++){
				k[i][j] = new DoubleMatrix(new double[3][4]);
			}
		}
		System.out.println(k[0][0].getRows());
//		System.out.println(v.getRange(1, 4, 2, 4));


		DoubleMatrix dd = new DoubleMatrix(new double[][] {
				{2, -2,1,-1},
				{-1,1,-2,2},
				{5,6,7,8}
		});

		DoubleMatrix rowvector = new DoubleMatrix(new double[][]{
				{-1.0,-1.0,-1.0}
		});

		DoubleMatrix colvector = new DoubleMatrix(new double[][]{
				{2.0},
				{2.0},
				{3.0}
		});




		System.out.println("-----------------------");
		System.out.println(dd.columnSums().transpose());
		System.out.println("----------2*2----4*4---------");
		System.out.println(rowvector.isRowVector() +"**"+colvector.isColumnVector());
		System.out.println(rowvector.rows+"**"+rowvector.columns);
		System.out.println("rowvec:"+rowvector );
		System.out.println(colvector.rows +"**"+colvector.columns);
		System.out.println("colvec"+colvector);
		System.out.println(dd.getRow(0)+"dd"+dd.getColumn(0)+" "+ dd.transpose().rows);
		System.out.println(dd.columnSums());
		System.out.println(dd.rowSums());
//		System.out.println(dd.mulRowVector(rowvector));
//		System.out.println(dd.mulColumnVector(colvector));

//		System.out.println(dd.mmul(v));
//		MatrixFunctions.expi(dd);
//		System.out.println(dd.mul(dd.gt(0)));
//		System.out.println("--------div---------------");
//		System.out.println(dd.div(2));
//		System.out.println("-----------rdiv------------");
//		System.out.println(dd.rdiv(2));
//		System.out.println("------------sub-----------");
//		System.out.println(dd.sub(2));
//		System.out.println("-----------rsub------------");
//		System.out.println(dd.rsub(2));

		//new DoubleMatrix();
		DoubleMatrix ffff = DoubleMatrix.zeros(4);
//		DoubleMatrix ffff = DoubleMatrix.eye(4);
		System.out.println("----------ffff-------------");
		//ffff.putColumn(1, dd);
		System.out.println(ffff);
		System.out.println(ffff.rows+"##"+ffff.columns);
		DoubleMatrix re = new DoubleMatrix(1,3,4,5,6);
		//dd.resize(3, 3);
//		System.out.println(ffff.sameSize(dd));
		System.out.println(re);

		int te = dd.columns/2;
		int pi = dd.rows/2;
		//dd.put(te, pi, re);
		System.out.println("----------lllll------------");
//		System.out.println(test(x,y));
//System.out.println(y);
//		for(int i=0; i< re.rows;i++){
//			for(int j=0; j<re.columns; j++){
//
//				dd.put(i+pi, j+te, re.get(i, j));
//			}
//		}

		DoubleMatrix gx = new DoubleMatrix(new double[][]{
				{-1.0,-1.0,-1.0,1.1},
				{1,3,4,-4},
				{2,5,6,-6},
				{3.2,7,8,-8},
				{4.4,9,10,-9}
		});
//		System.out.println(gx.addRowVector(colvector));
//		System.out.println(gx.addRowVector(rowvector));
//		System.out.println(colvector);
//		dd.copy(re);
//		System.out.println(dd);
		System.out.println(x);
		System.out.println(y);
		test(x,y);
		System.out.println(x);
		System.out.println(y);
		System.out.println("++++----");

		DoubleMatrix rindex = new DoubleMatrix(new double[][]{
				{1,2},
				{1,2}
		});
		DoubleMatrix cindex = new DoubleMatrix(new double[][]{
				{1,0},
				{4,2}
		});


		System.out.println(gx.gt(4));
//		System.out.println(rindex.eqi(cindex));
//		rindex = Jblas_util.put(gx, y, 2, 1);
		System.out.println(gx.getRange(0, 2, 2, 4));
//		System.out.println(Jblas_util.zeropadding(y, 2));

	}

	static DoubleMatrix test(DoubleMatrix p, DoubleMatrix k){
		p.addi(k);
		System.out.println("p");
		//y= y.add(5);
		System.out.println( p );
		k.addi(110);
		return p;
	}
}
