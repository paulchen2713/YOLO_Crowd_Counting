//Optimal Parking A1065141 
#include <iostream>
#include <algorithm>

using namespace std;

int main() { 
    int test, stores = 0;
	int a[100] = {0}; //register for store's location
	
    cin >> test; //given the number of test cases
    while(test--) {
        cin >> stores; //given the number of stores must visit
        for (int i = 0; i < stores; i++) {
        	cin >> a[i];
		}
	} 
		/* where the actual parking place is unimportant, as long as he parked between the 
		closest store and the farthest store that he wanna visit. the minimal travel distance 
		required would be the same = (max - min)x2 */
        sort (a, a + stores); 
        cout << "The minimal distance required is: " << (a[stores-1] - a[0]) *2 << endl;
    
    system("pause");
    return 0;
}

