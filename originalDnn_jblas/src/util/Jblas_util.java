package util;

import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.exceptions.SizeException;

public class Jblas_util {

	public static void checkRows(int x, int r) {
		if (x < r) {
			throw new SizeException(
					"Matrix does not have the necessary number of rows ("
							+ x + " != " + r + ").");
		}
	}

	public static void checkColumns(int x, int c) {
		if (x < c) {
			throw new SizeException(
					"Matrix does not have the necessary number of columns ("
							+ x + " != " + c + ").");
		}
	}

	/**
	 * vをxに(row,column)(v.rows,v.columns)まで代入する。v(row+v.rows,column+v.columns)はxのrows,columnsより小さいこと
	 * returnしているがputのためxとして渡したMatrix自体が書き換わる。書き換えたくないならx.dup()で渡すこと
	 * @param x 代入される行列
	 * @param v 代入する値
	 * @return 置き換えた結果
	 */
	public static DoubleMatrix put(DoubleMatrix x, DoubleMatrix v, int row, int column) {

		checkRows(x.rows,v.rows+row);
		checkColumns(x.columns,v.columns+column);
		for(int i=row; i<v.rows+row; i++){
			for(int j=column; j<v.columns+column; j++){
//				System.out.println(v.get(i-row,j-column));
				x.put(i, j, v.get(i-row,j-column));
			}
		}
		return x;
	}

	/**
	 * vをxに(row,column)(v.rows,v.columns)まで代入する。v(row+v.rows,column+v.columns)はxのrows,columnsより小さいこと
	 * returnしているがputのためxとして渡したMatrix自体が書き換わる。書き換えたくないならx.dup()で渡すこと
	 * @param x 代入される行列
	 * @param v 代入する値
	 * @return 置き換えた結果
	 */
	public static FloatMatrix put(FloatMatrix x, FloatMatrix v, int row, int column) {

		checkRows(x.rows,v.rows+row);
		checkColumns(x.columns,v.columns+column);
		for(int i=row; i<v.rows+row; i++){
			for(int j=column; j<v.columns+column; j++){

				x.put(i, j, v.get(i-row,j-column));
			}
		}
		return x;
	}


	public static FloatMatrix addRange(FloatMatrix x, FloatMatrix v, int row, int column){
		FloatMatrix y = x.dup();
		return addiRange(y,v,row,column);
	}

	public static FloatMatrix addiRange(FloatMatrix x, FloatMatrix v, int row, int column){
		checkRows(x.rows,v.rows+row);
		checkColumns(x.columns,v.columns+column);

		FloatMatrix y = x.getRange(row, row+v.rows, column, v.columns);
		y.addi(v);

		return put(x,y,row,column);
	}

	/**
	 * ゼロパディングを行う。行った行列を返す
	 * @param x ゼロパティングを行う行列
	 * @param padding ゼロパディングを行う回数。両端に行うため*2となる。
	 * @return 行った結果
	 */
	public static DoubleMatrix zeropadding(DoubleMatrix x, int padding){
		DoubleMatrix zero = DoubleMatrix.zeros(x.rows+(2*padding), x.columns+(2*padding));
		put(zero, x, padding, padding);
		return zero;
	}

	/**
	 * ゼロパディングを行う。行った行列を返す
	 * @param x ゼロパティングを行う行列
	 * @param padding ゼロパディングを行う回数。両端に行うため*2となる。
	 * @return 行った結果
	 */
	public static FloatMatrix zeropadding(FloatMatrix x, int padding){
		FloatMatrix zero = FloatMatrix.zeros(x.rows+(2*padding), x.columns+(2*padding));
		put(zero, x, padding, padding);
		return zero;
	}

	/**
	 * xとvの比較を行う。vの要素を一つずつ取出しxの要素と比較する。どちらが大きくてもよいが、
	 * 結果はxと同じ大きさで返ってくる
	 * @param x 比較対象
	 * @param v xと比較を行いたい行列
	 * @return 結果（すべて0,1。1がマッチした要素）
	 */
	public static FloatMatrix element_eq(FloatMatrix x, FloatMatrix v){
		FloatMatrix result = new FloatMatrix(x.rows,x.columns);
//		FloatMatrix y = result.dup();
		for(int i=0; i<v.rows; i++){
			for(int j=0; j<v.columns; j++){
				if(v.get(i,j) != 0){
					result.addi(x.eq(v.get(i,j)));
				}
			}
		}

		return result.gt(0);
	}


}
