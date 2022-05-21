#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>

using namespace std;
 
int main() {
	//先建立檔案資料，並讀取檔案
    fstream file1;
    file1.open("list_0.txt");
    
    //判斷是否有讀到檔案
    if (file1 != NULL) {
	    #define point_size 100
    	
    	//建立字串陣列
		string ss[point_size];
		for (int i = 0; i < point_size; i++) {
			char ii[50];
			itoa(i + 1, ii, 10); //int 轉成 char
		}
	    //配置記憶體，建立動態陣列
		int *count = NULL;
		count = new int[point_size];
		for (int i = 0; i < point_size; i++) {
			count[i] = 0; //初始化陣列為0
		}
		
		#define size 100
	    char oneline[size];
    	bool flag = true;
    	
    	//讀取檔案
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
    	
    	//將分組結果輸出並列印出結果
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
	    cout << "內容已經儲存並輸出成檔案\"result_0.txt\"，存於同一資料夾目錄內\n\n";
	
	int i,t;
    double x[4],y[4],ax,ay;
     
    while(cin>>x[0]>>y[0])//輸入測資 
    {
        ax=x[0];//全部的點相加 
        ay=y[0];//全部的點相加 
         
        for(i=1;i<4;i++)//輸入測資
        {
            cin>>x[i]>>y[i]; 
            ax+=x[i];//全部點相加 
            ay+=y[i];//全部點相加 
        }
         
        for(i=0;i<4;i++)
        {
            for(t=i+1;t<4;t++)
            {
                if(x[i]==x[t]&&y[i]==y[t])//找到一樣的點 
                {
                    cout<<fixed<<setprecision(3);//精確到小數點下三位 
                    cout<<ax-3*x[i]<<" ";//輸出x座標 
                    cout<<ay-3*y[i]<<endl;//輸出y座標 
                }
            }
        }
    }
    
    } else {
    	cout << "檔案讀取失敗，請將檔案名設為\"list_0\"，\n並將內容寫成一行一個名字，如下\n\n";
    	cout <<"list_1.txt\n--------------------------------------------\n";
    	cout <<"AAA\nBBB\nCCC\nDDD\nEEE\nFFF\nGGG\nHHH\n.....\n....\n...\n\n\n\n--------------------------------------------\n\n\n\n";
	}
    return 0;
}

