// #include <iostream>
// #include <string>
// using namespace std;
// int main() {
//   int i = 0, j = 0;
//   char str[10], e;
//   cin >> str;
//   cin >> e;
//   while (str[i] != '\0') {
//     if (str[i] != e) {
//       str[j++] = str[i];
//     }
//     i++;
//   }
//   str[j] = '\0';

//   cout << str;

//   return 0;
// }
#include <iostream>
using namespace std;
int main() {
  int n1, n2;
  int *p1 = &n1, *p2 = &n2, *p;
  cout << "Input the first number : " << endl;
  cin >> n1;
  cout << "Input the second number : " << endl;
  cin >> n2;
  if (*p1 > *p2) {
    p = p1;
    p1 = p2;
    p2 = p;
  }
  
  cout << *p1 << *p2 << endl;
}