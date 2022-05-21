//Uva. 11417 - GCD 
//find the value of GCD
//keyconcept. define GCD function on our own
#include<iostream>

using namespace std;

int gcd (int n, int m) { //define GCD function
    if (m == 0) { //cache for 0 special case
    	return n;
	}
	else {
		return gcd(m, n%m);
	}
}

int main() {
    int n; 
    int sum; //store final answer
    while (cin >> n && n != 0) {
        sum = 0;
        for (int i = 1; i < n; i++) {
        	for (int j = i+1; j <= n; j++) {
        		//cout << gcd(i, j) << endl;
        		sum += gcd(i, j);
			}
		}
    cout << sum << endl;
    }
    return 0;
}

