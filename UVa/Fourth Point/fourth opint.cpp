//find the Fourth Point of the parallelogram A1065141  
#include <iostream>
#include <iomanip> //function_precision

using namespace std;

int main() {  
    double x1, y1, x2, y2, x3, y3, x4, y4; //(x1, y1) (x2, y2) (x3, y3) (x4, y4)
    
    /*����Ū�J4��(x, y)�y���I�A�N��1��redundant data�A�]���u��3���I�N��w�@�X����|��ΡA
	�����N3�I�w�X������|��ΫD�ߤ@�ѦӦ�3�إi��A�ҥH�D�ج�²�ƥu�ݨD�X���Ʈy�Ъ���*/ 
    while(cin >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4) {
        cout << fixed << setprecision(3); //setting significant digits with 3
        
        /*��y�Э��ƮɫO�d����Ǹ��p���C�Ҽ{����|���1234�b�w���@�u���p�U��4�إi��ѡA���O��:
		���]�H1����ǡA�i�q�F��12, �﨤�u13, �F��14, �ι���24�e�X�@�ߤ@����|���*/ 
        if (x3 == x4 && y3 == y4)
        	cout << (x1 + x2) - x3 << " " << (y1 + y2) - y3 << endl;
        else if (x2 == x4 && y2 == y4)
        	cout << (x1 + x3) - x2 << " " << (y1 + y3) - y2 << endl; 
        else if (x2 == x3 && y2 == y3)
        	cout << (x1 + x4) - x2 << " " << (y1 + y4) - y2 << endl; 
        else if (x1 == x3 && y1 == y3)  
            cout << (x2 + x4) - x1 << " " << (y2 + y4) - y1 << endl;
        else
        	cout << "��J���꦳�~" << endl;
    }
    return 0;
}

