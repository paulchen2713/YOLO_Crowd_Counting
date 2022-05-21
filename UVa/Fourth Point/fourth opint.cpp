//find the Fourth Point of the parallelogram A1065141  
#include <iostream>
#include <iomanip> //function_precision

using namespace std;

int main() {  
    double x1, y1, x2, y2, x3, y3, x4, y4; //(x1, y1) (x2, y2) (x3, y3) (x4, y4)
    
    /*測資讀入4個(x, y)座標點，代表有1個redundant data，因為只需3個點就能定一出平行四邊形，
	但任意3點定出的平行四邊形非唯一解而有3種可能，所以題目為簡化只需求出重複座標的解*/ 
    while(cin >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4) {
        cout << fixed << setprecision(3); //setting significant digits with 3
        
        /*當座標重複時保留測資序號小的。考慮平行四邊形1234在已知一線情況下有4種可能解，分別為:
		假設以1為基準，可從鄰邊12, 對角線13, 鄰邊14, 或對邊24畫出一唯一平行四邊形*/ 
        if (x3 == x4 && y3 == y4)
        	cout << (x1 + x2) - x3 << " " << (y1 + y2) - y3 << endl;
        else if (x2 == x4 && y2 == y4)
        	cout << (x1 + x3) - x2 << " " << (y1 + y3) - y2 << endl; 
        else if (x2 == x3 && y2 == y3)
        	cout << (x1 + x4) - x2 << " " << (y1 + y4) - y2 << endl; 
        else if (x1 == x3 && y1 == y3)  
            cout << (x2 + x4) - x1 << " " << (y2 + y4) - y1 << endl;
        else
        	cout << "輸入測資有誤" << endl;
    }
    return 0;
}

