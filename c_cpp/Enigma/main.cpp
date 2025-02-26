#include<iostream>
#include<string>
#include <fstream>
class Enigma {
public:
    char alphabet[26]={'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',};

	void input(const std::string k) {
        c=k;
    }

    char char_change_int(char k){
            for(int m=0;m<26;m++){
                if(k==alphabet[m]){
                return m; 
                }
            }
    }
	std::string encode() {
        int count=0;
        std::string m;
        for(int i=0;i<c.size();i++){
            char temp;
            temp=char_change_int(c[i]);
            for(int i=0;i<5;i++){
                temp+=rotator_code[i];
                if(temp>=26){
                    temp-=26;
                }
            }

            temp=27-temp;
            for(int i=0;i<5;i++){
                temp+=rotator_code[4-i];
                if(temp>=26){
                    temp-=26;
                }
            }
            count++;
            for(int i=0;i<5;i++){
                if(count==rotator_ch[i]){
                    rotator_code[i]+=1;
                }
            }
            temp=temp+97;
            std::cout<<(char)temp;
            m+=(char)temp;
            
        }
        std::cout<<std::endl;
        return m;
	}
private:
	std::string c;
	int count_rotator_ch = 0;
	int rotator_code[5] = { 1,3,4,5,2 };
	const int rotator_ch[5] = { 2,9,12,7,5 };

};

int main() {
	Enigma t1;
    std::fstream code,decode;
    code.open("/home/ccy/workspace/c_cpp/Enigma/decode.txt",std::ios::in);
    std::string k1;
    std::string k2;
   
    if(!code.is_open()){
        std::cout<<"open fail"<<'\n';
        return 0;
    };
  while (std::getline(code,k1)) {
    	t1.input(k1);
        k2+=t1.encode();
  }
  code.close();
  decode.open("/home/ccy/workspace/c_cpp/Enigma/code.txt",std::ios::out);
  decode<<k2;
 
  decode.close();
	return 0;
}
