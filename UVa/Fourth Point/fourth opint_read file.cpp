#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

using namespace std;
 
int main() {
	//���إ��ɮ׸�ơA��Ū���ɮ�
    fstream file1;
    file1.open("list_0.txt");
    
    //�P�_�O�_��Ū���ɮ�
    if (file1 != NULL) {
	    #define point_size 100
    	
    	//�إߦr��}�C
		string ss[point_size];
		for (int i = 0; i < point_size; i++) {
			char ii[50];
			itoa(i + 1, ii, 10); //int �ন char
		}
	    //�t�m�O����A�إ߰ʺA�}�C
		int *count = NULL;
		count = new int[point_size];
		for (int i = 0; i < point_size; i++) {
			count[i] = 0; //��l�ư}�C��0
		}
		
		#define size 100
	    char oneline[size];
    	bool flag = true;
    	
    	//Ū���ɮ�
        while (file1.getline(oneline, sizeof(oneline), '\n')) {
        	flag = true;
        	while (flag) {
        		if (count[i] != team_member_count) {
        			count[i]++;
    	    		ss[i] = ss[i] + oneline + ", ";
        			flag = false;
	    		}
	    	}
    	}
    	
    	//�N���յ��G��X�æC�L�X���G
        ofstream file2;
        file2.open("result_00.txt");
        cout << "--------------------------------------------\n";
        for (int j = 0; j < team_count; j++) {
        	cout << ss[j] << "\n\n";
        	file2 << ss[j] << "\n\n";
    	}
	
	    file1.close();
    	file2.close();
	
	    cout << "--------------------------------------------\n";
	    cout << "���e�w�g�x�s�ÿ�X���ɮ�\"result_0.txt\"�A�s��P�@��Ƨ��ؿ���\n\n";
	
	int i,t;
    double x[4],y[4],ax,ay;
     
    while(cin>>x[0]>>y[0])//��J���� 
    {
        ax=x[0];//�������I�ۥ[ 
        ay=y[0];//�������I�ۥ[ 
         
        for(i=1;i<4;i++)//��J����
        {
            cin>>x[i]>>y[i]; 
            ax+=x[i];//�����I�ۥ[ 
            ay+=y[i];//�����I�ۥ[ 
        }
         
        for(i=0;i<4;i++)
        {
            for(t=i+1;t<4;t++)
            {
                if(x[i]==x[t]&&y[i]==y[t])//���@�˪��I 
                {
                    cout<<fixed<<setprecision(3);//��T��p���I�U�T�� 
                    cout<<ax-3*x[i]<<" ";//��Xx�y�� 
                    cout<<ay-3*y[i]<<endl;//��Xy�y�� 
                }
            }
        }
    }
    
    } else {
    	cout << "�ɮ�Ū�����ѡA�бN�ɮצW�]��\"list_0\"�A\n�ñN���e�g���@��@�ӦW�r�A�p�U\n\n";
    	cout <<"list_1.txt\n--------------------------------------------\n";
    	cout <<"AAA\nBBB\nCCC\nDDD\nEEE\nFFF\nGGG\nHHH\n.....\n....\n...\n\n\n\n--------------------------------------------\n\n\n\n";
	}
    return 0;
}

