#include <stdio.h>
int main ()
{
   int i, j, a = 0;
   for (i = 0; i < 2; i++)
   {  for (j = 1; j < 5; j++, a++)
      {  if (!(j % 3)) break;
          a++;
          printf("a = %d\n", a++);
      }
   }
   for (i = 0; i < 2; i++)
   {  for (j = 0; j < 4; j++, a++)
      {  if (j % 2) continue;
         a++;
      }
      printf("a = %d\n", ++a);
   }
}
