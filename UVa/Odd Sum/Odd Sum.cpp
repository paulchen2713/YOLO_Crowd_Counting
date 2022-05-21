//Uva.10783 - Odd Sum
//find the odd sum of a certain range 
#include<iostream>

using namespace std;

/* 2
   [1, 5]          [3, 5]
   1 + 3 + 5 = 9,  3 + 5 = 8 */

int main() {
    int n;     //number of testing data
	int a, b;  //a. lowerbound, b. upperbound
	int sum;   //store final answer
	
    cin >> n; 
    for(int i = 0; i < n; i++){
        cin >> a >> b;
        sum = 0;
        for(int j = a; j <= b; j++){ //searching range [a, b]
            if ((j % 2) == 1) { //finding odd digits
                sum = sum + j;
            }
        }
        cout << "Case " << i + 1 <<": "<< sum << endl;
    }
    //system("pause");
}

