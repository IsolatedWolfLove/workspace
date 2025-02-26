#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int user();
int manager();
int menu();

int n,n2;									//n为当前学生系统的人数、n2为当前图书系统的书本数

struct Student
{
	char ID[10];							//学号 
	int Age;								//年龄
	char Name[10];							//姓名 
	char Sex;								//性别
	char Major[20];							//专业
	float grade[3];							//高数、大英、编程成绩
}stu[100];
int len1=sizeof(struct Student); 

int present()								//浏览学生信息 
{
	int i;
	if(n>0)
	{
		for(i=0;i<n;i++)
		{
			printf("\t\t\t学号：%s   ",stu[i].ID);
			printf("姓名：%s   ",stu[i].Name);
			printf("性别：%c   ",stu[i].Sex);
			printf("年龄：%d   ",stu[i].Age);
			printf("专业：%s   ",stu[i].Major);
			printf("\n");
		}
		system("pause");
		return 1;	
	}
	else
	{
		printf("\t\t\t\t无学生信息\n");
		system("pause");
		return 0;	
	} 
}


void user_input()							//添加学生信息
{
	int m,j,g;								//m为需要添加的学生人数
	printf("\t\t\t\t请输入你要添加的人数：");
	scanf("%d",&m);
	int i=n;
	if(m>0)
	{
		while(i<n+m)
		{
		printf("\t\t\t\t请输入第%d位学生的信息\n",i+1);
			
flag:		printf("\t\t\t\t请输入学号："); 
			scanf("%s",stu[i].ID); 
			getchar();
			
//----------判断学号有无重复----------
			int j;
			for(j=0;j<i;j++)
			if(strcmp(stu[i].ID,stu[j].ID)==0) 
			{
				printf("\t\t\t\t学号已重复,请重新输入\n ");
				goto flag;
			}
//----------判断学号有无重复----------			
			
			printf("\t\t\t\t请输入姓名：");
			scanf("%s",stu[i].Name); 
			getchar(); 
		
			printf("\t\t\t\t请输入性别F（女）或M（男）:"); 
			scanf("%c",&stu[i].Sex); 
			getchar(); 
			
			printf("\t\t\t\t请输入年龄："); 
			scanf("%d",&stu[i].Age); 
			
			printf("\t\t\t\t请输入专业：");
			scanf("%s",stu[i].Major);
			
			FILE *p=fopen("学生信息.txt","ab+");
			if(p==NULL)
			exit(0); 
			fwrite(&stu[i],len1,1,p);
			fclose(p);
			i++;
		}
		printf("\t\t\t\t学生信息已经添加！！！\n"); 
			printf("\t\t\t------------------------------------------------------------------------------------------\n");
			for(i=n;i<n+m;i++)
			{
				printf("\t\t\t学号：%s \t",stu[i].ID);
				printf("姓名：%s \t",stu[i].Name);
				printf("性别：%c \t",stu[i].Sex);
				printf("年龄：%d \t",stu[i].Age);
				printf("专业：%s \t",stu[i].Major);
				printf("\n");
			}
			
			printf("\t\t\t------------------------------------------------------------------------------------------\n");
	}
	n+=m;
	system("pause");
}

void user_revise()							//修改学生信息 
{	
	//更新：修改完的信息覆盖掉文件中的信息 
	char ID[10];
	if(present())
	{
	while(1)
	{
		printf("\t\t\t\t请问你需要修改学生信息的学号是：");
		scanf("%s",ID);
		getchar(); 
		int i,j,flag=0;
		for(i=0;i<n;i++)
		{
			if(strcmp(ID,stu[i].ID)==0)
			{
			j=i;							//记录找到的学生信息下标 
			flag=1;							//找到了 
			break;
			}
		}
		if(flag==1)							//开始修改 
		{	while(1)
			{
				printf("\t\t\t\t请选择修改选项或退出：1.学号2.姓名3.性别4.年龄5.专业 0.退出\n");
				int x; 
				scanf("%d",&x);
				switch(x)
				{
				case 1:
					{
					printf("\t\t\t\t请输入学号：");
					scanf("%s",stu[j].ID);
					printf("\t\t\t\t修改成功！");
					}
				case 2:
					{
					printf("\t\t\t\t请输入新名字：\n");
					scanf("%s",stu[j].Name);
					printf("\t\t\t\t修改成功！");
					break;	
					}
				case 3:
					{
					printf("\t\t\t\t请修改性别");
					scanf("%c",&stu[j].Sex);
					getchar(); 
					printf("\t\t\t\t修改成功！");
					break;
					}	 
				case 4:
					{
					int age;
					printf("\t\t\t\t请输入新年龄：\n");
					scanf("%d",&age);
					stu[j].Age=age;
					printf("\t\t\t\t修改成功！");
					break;	
					}
					
					break; 
				case 5:
					{
					printf("\t\t\t\t请输入新专业：");
					scanf("%s",stu[j].Major);
					printf("\t\t\t\t修改成功！");
					break;	
					}
				
				case 0:
					{
						//开始查找文件中的要修改的文件,将控制台修改完的信息覆盖到文件中要修改的信息
						FILE *p=fopen("学生信息.txt","r+b");	
						if(p==NULL)
						{
							exit(0);	
						}		
						fseek(p,j*len1,SEEK_SET);			//把指针移向到要文件中要修改的信息 
						fwrite(&stu[j],len1,1,p); 
						fclose(p);
						return;
					}
			
				}
				printf("\t\t\t学号：%s\t ",stu[i].ID);
				printf("姓名：%s\t ",stu[i].Name);
				printf("性别：%c\t ",stu[i].Sex);
				printf("年龄：%d\t ",stu[i].Age);
				printf("专业：%s\t ",stu[i].Major);
				printf("\n");		
			}
			
		} 
		else
		{
			printf("\t\t\t\t找不到该学号的学生信息\n"); 
			break;
		}	
	}
	system("pause");
	}
}

void deletee()								//删除学生信息
{
	//更新：删除后把文件清空重新保存 
	int i,flag1;
	char flag;
	char ID1[10]; 
	if(n!=0)
	{
		for(i=0;i<n;i++)
		{
		printf("\t\t\t学号：%s ",stu[i].ID);
		printf("姓名：%s ",stu[i].Name);
		printf("性别：%c ",stu[i].Sex);
		printf("年龄：%d ",stu[i].Age);
		printf("专业：%s",stu[i].Major);
		printf("\n");
		}
		while(1)
		{
		printf("\t\t\t\t请选择你要删除的信息（填学号）：");
		scanf("%s",ID1);
		for(i=0;i<n;i++)
		{
			if(strcmp(ID1,stu[i].ID)==0)
			{
				printf("\t\t\t\t找到了！\n");
				printf("\t\t\t\t真的要删除？（Y/N）\n");
				scanf("%c",&flag);
				switch(flag)
				{
				case 'Y':
				{
					int j;
					//memset(&stu[i],0,sizeof(struct student));
					
					for(j=i;j<n;j++)//删除的信息后面的信息往前一位，要删除的信息被前一信息覆盖,即删除 
					{
						stu[j]=stu[j+1];
					}
					n-=1;
					FILE *p=fopen("学生信息.txt","wb");
					if(p==NULL)
					exit(0);
					for(i=0;i<n;i++)
					{
						if(fwrite(&stu[i],len1,1,p)!=1) 
						{
						printf("\t\t\t\t不能保存\n");	
						getchar(); 
						}
						
					}
					fclose(p);
				break;
				}
				case 'N':flag1=1;break;
				}	
			}
		}
		if(flag1==1)
		break;
		
		printf("\t\t\t\t找不到！\n");	
	}
		
	}
	
	else
		printf("\t\t\t\t无信息可删除！\n");
	system("pause");
}

void find()//查找
{
	char ID[10];
	int i,re=0,flag=0;
	char x;
	
	while(1)
	{	printf("\t\t\t\t请输入你要查找学生信息的学号:");
		scanf("%s",ID);
		getchar(); 
		for(i=0;i<n;i++)
		{
			if(strcmp(ID,stu[i].ID)==0)
			{
			printf("\t\t\t\t找到了！！\n");
			printf("\t\t\t学号：%s\t ",stu[i].ID);
			printf("姓名：%s\t ",stu[i].Name);
			printf("性别：%c\t ",stu[i].Sex);
			printf("年龄：%d\t ",stu[i].Age);
			printf("专业: %s\t ",stu[i].Major);
			printf("\n");
			flag=1;
			break;
			}
		} 
		if(flag!=1)
		printf("\t\t\t\t无该学号的学生信息！\n");
		
		printf("\t\t\t\t是否继续查找（Y/N）"); 
		scanf("%c",&x);
		switch(x)
		{
		case 'Y':break;
		case 'N':re=1;break;	
		}
 
		if(re==1)
		break;
	}
	system("pause");
 } 
 
 void enter()//清空系统 
{
	FILE *p=fopen("学生信息.txt","w");
	if(p==NULL)
	exit(0);
	fclose(p);	
} 

struct Book
{
	char title[50];							//书名
	int num1;								//在馆数量
	char author[20];						//作者
	int num2;								//收入数量
}book[100];
int len2=sizeof(struct Book); 

void lib_present()							//浏览图书信息 
{
	int i,j;
	if(n2>0)
	{
		for(i=0;i<n2;i++)
		{
			printf("\t\t\t编号：%d   ",i+1);
			printf("书名：%s   ",book[i].title);
			printf("作者：%s   ",book[i].author);
			printf("在馆数量：%d   ",book[i].num1);
			printf("收入数量：%d   ",book[i].num2);
			printf("\n");
		}
		system("pause");	
	}
	else
	{
		printf("\t\t\t\t无图书信息\n");
		system("pause");	
	} 
}

void lib_add()								//添加图书
{
	int m,j,g,k;							//m为需要添加的不同书本数
	printf("\t\t\t\t请输入你要添加的不同书本数：");
	scanf("%d",&m);
	int i=n2;
	char tt[50];
		
	if(m>0)
	{
		while(i<n2+m)
		{
		if(i==0)
		{
			printf("\t\t\t\t请输入第%d本书的信息\n",i+1);
			printf("\t\t\t\t请输入书名：");
			scanf("%s",book[i].title);
			getchar();
			printf("\t\t\t\t请输入作者：");
			scanf("%s",book[i].author); 
			getchar(); 
			printf("\t\t\t\t请输入收入该书的数量： ");
			scanf("%d",&k); 
			book[i].num2=k;
			book[i].num1=book[i].num2;
			FILE *q=fopen("图书信息.txt","ab+");
			if(q==NULL)
			exit(0); 
			fwrite(&book[i],len2,1,q);
			fclose(q);
			i++;
		}
		else
		{
		printf("\t\t\t\t请输入第%d本书的信息\n",i+1);
		printf("\t\t\t\t请输入书名："); 
		scanf("%s",tt); 
		
		int j;
		for(j=0;j<i;j++)
		{
			if(strcmp(tt,book[j].title)!=0) 
			{
				strcpy(book[i].title,tt);
				printf("\t\t\t\t请输入作者：");
				scanf("%s",book[i].author); 
				getchar(); 
				printf("\t\t\t\t请输入收入该书的数量： ");
				scanf("%d",&k); 
				book[i].num2=k;
				book[i].num1=book[i].num2;
			}
			if(strcmp(tt,book[j].title)==0) 
			{
				printf("\t\t\t\t请输入收入该书的数量： ");
				scanf("%d",&k); 
				book[i].num2+=k;
				book[i].num1+=k;
			}
		
		}
			
			FILE *q=fopen("图书信息.txt","ab+");
			if(q==NULL)
			exit(0); 
			fwrite(&book[i],len2,1,q);
			fclose(q);
			i++;
		}
		}
		printf("\t\t\t\t图书信息已经添加！！！\n"); 
			printf("\t\t\t------------------------------------------------------------------------------------------\n");
			for(i=n2;i<n2+m;i++)
			{
				printf("\t\t\t编号：%d   ",i+1);
				printf("书名：%s   ",book[i].title);
				printf("作者：%s   ",book[i].author);
				printf("在馆数量：%d   ",book[i].num1);
				printf("收入数量：%d   ",book[i].num2);
				printf("\n");
			}
			
			printf("\t\t\t------------------------------------------------------------------------------------------\n");
	}
	n2+=m;
	system("pause");
}

void research_title()							//用书名查找
{
	int i,re=0,flag=0;
	char x;
	char title[50];
	while(1)
	{
		printf("\t\t\t\t请输入你要查找书的名字:");
		scanf("%s",title);
		getchar(); 
		for(i=0;i<n2;i++)
		{
			if(strcmp(title,book[i].title)==0)
			{
			printf("\t\t\t\t找到了！！\n");
			printf("\t\t\t书名：%s   ",book[i].title);
			printf("作者：%s   ",book[i].author);
			printf("在馆数量：%d   ",book[i].num1);
			printf("收入数量：%d   ",book[i].num2);
			printf("\n");
			flag=1;
			break;
			}
		} 
		if(flag!=1)
		printf("\t\t\t\t无该书的信息！\n");
		
		printf("\t\t\t\t是否继续查找（Y/N）"); 
		scanf("%c",&x);
		switch(x)
		{
		case 'Y':break;
		case 'N':re=1;break;	
		}
 
		if(re==1)
			break;
	}
	system("pause");
}

void research_author()						//用作者查找
{
	int i,re=0,flag=0;
	char x;
	char author[20];
	while(1)
	{
		printf("\t\t\t\t请输入你要查找书的作者:");
		scanf("%s",author);
		getchar(); 
		for(i=0;i<n2;i++)
		{
			if(strcmp(author,book[i].author)==0)
			{
			printf("\t\t\t\t找到了！！\n");
			printf("\t\t\t书名：%s   ",book[i].title);
			printf("作者：%s   ",book[i].author);
			printf("在馆数量：%d   ",book[i].num1);
			printf("收入数量：%d   ",book[i].num2);
			printf("\n");
			flag=1;
			break;
			}
		} 
		if(flag!=1)
		printf("\t\t\t\t无该书的信息！\n");
		
		printf("\t\t\t\t是否继续查找（1/0）"); 
		scanf("%c",&x);
		switch(x)
		{
		case 'Y':break;
		case 'N':re=1;break;	
		}
 
		if(re==1)
			break;
	}
	system("pause");
}

void lib_research()							//图书查找
{
	int i;
	while(1)
	{
		printf("\t\t\t\t请选择你要查找书的方式（1.书名 2.作者）");
		scanf("%d",&i);
		switch(i)
		{case 1:research_title();break;
		case 2:research_author();break;
		default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
}

void lib_borrow()							//借书
{
	char title[50];
	printf("\t\t\t\t请输入你要借阅书的名字（一次借一本）：");
	scanf("%s",title);
	getchar(); 
	for(int i=0;i<n2;i++)
	{
		if(strcmp(title,book[i].title)==0)
		{
			if(book[i].num1>0)
			{
			book[i].num1--;
			printf("\t\t\t\t借阅成功！");
			}
			else
			printf("\t\t\t\t该书已全被借出，借阅失败\n");
		}
		else
			printf("\t\t\t\t没有该书，借阅失败\n");
	}
	system("pause");
}
void lib_return()							//还书
{
	char title[50];
	printf("\t\t\t\t请输入你要归还书的名字（一次还一本）：");
	scanf("%s",title);
	getchar(); 
	for(int i=0;i<n2;i++)
	{
		if(strcmp(title,book[i].title)==0)
		{
			if(book[i].num2>=book[i].num1)
			{
			book[i].num1++;
			printf("\t\t\t\t归还成功！");
			}
			else
			printf("\t\t\t\t该书不属于本图书馆\n");
		}
		else
			printf("\t\t\t\t该书不属于本图书馆\n");
	}
	system("pause");
}

void lib_recommend()						//好书推荐
{
	printf("  《平凡的世界》--路遥：这是一部描绘中国农村家庭三代人生活的长篇小说，展现了人们在困境中如何坚持并寻找希望\n\n");
	printf("  《追风筝的人》--卡勒德·胡塞尼：这是一部描述阿富汗历史和文化变迁的小说，同时也是一部关于友情、背叛和救赎的故事\n\n");
	printf("  《活着》--余华： 这是一部描绘普通人生活经历的小说，以其深刻的人生哲理和对人性的独到见解而广受好评\n\n");
	printf("  《云边有个小卖部》--张嘉佳：这是一部充满人情味和温馨的小说，讲述了一个小村庄里的人们如何在困难中相互扶持\n\n");
	printf("  《解忧杂货店》--日东野圭吾：这是一部关于心理咨询的小说，通过一家杂货店解决人们心中的疑惑和困扰\n\n");
	printf("  《傲慢与偏见》--简·奥斯汀：讲述了伊丽莎白·班内特和达西先生之间的爱情故事，以及他们在社会阶级和个人偏见面前的挑战\n\n");
	printf("  《老人与海》--埃内斯特·海明威：讲述了一个老渔夫桑地亚哥与他的大马林鱼搏斗的故事，以及他在孤独和挫折中坚持的精神\n\n");
	printf("  《罪与罚》--陀思妥耶夫斯基：讲述了主人公罗德里克·罗斯科尔尼科夫在犯罪后的内心挣扎和救赎之路\n\n");
	printf("  《人类简史》--尤瓦尔·赫拉利：这本书以简洁明了的语言，概述了人类历史的发展和演变\n\n");
	printf("  《百年孤独》--加西亚·马尔克斯：这本拉丁美洲魔幻现实主义的经典之作，讲述了布恩迪亚家族七代人的故事\n\n");
	printf("  《小王子》--圣埃克苏佩里：这是一本适合所有年龄段读者的童话故事，讲述了一个小王子从星球上旅行到地球的经历\n\n");
	printf("  《资本论》--卡尔·马克思：这是一部经典的政治经济学著作，对资本主义经济体系进行了深入的分析和批判\n\n");
	printf("  《我与地坛》--史铁生：如果你对人生有所怀疑，那就去读读这本书，用心感受你周围的爱\n\n");
	system("pause");
	system("cls");
}

void eat()									//饮食
{
	printf("\t\t\t\t中南大学共有12个食堂，分布如下:\n");
	printf("\t\t\t\t┌───────────────────────────────────────────────────┐\n");  
	printf("\t\t\t\t│   南校区   │ 二食堂（升华餐饮楼），七食堂，八食堂 │\n");
	printf("\t\t\t\t│───────────────────────────────────────────────────│\n");
	printf("\t\t\t\t│   新校区   │ 一食堂，中南讲堂一楼食堂             │\n");
	printf("\t\t\t\t│───────────────────────────────────────────────────│\n");
	printf("\t\t\t\t│   校本部   │ 三食堂，五食堂，九食堂（教工餐饮楼） │\n");
	printf("\t\t\t\t│───────────────────────────────────────────────────│\n");
	printf("\t\t\t\t│  铁道校区  │ 四食堂，六食堂                       │\n");
	printf("\t\t\t\t│───────────────────────────────────────────────────│\n");
	printf("\t\t\t\t│  湘雅校区  │ 十食堂                               │\n");
	printf("\t\t\t\t│───────────────────────────────────────────────────│\n");
	printf("\t\t\t\t│ 湘雅新校区 │ 十一食堂，十二食堂                   │\n");
	printf("\t\t\t\t└───────────────────────────────────────────────────┘\n");	
	printf("\t\t\t\t也推荐住升华公寓的同学们去后湖小区吃饭\n");
	printf("\t\t\t\t麓山南路的餐馆是团建的好地方呦\n");
	system("pause");
	system("cls");
}

void health()								//医疗
{
	printf("\t\t\t\t中南大学的医疗救治地址如下:\n");
	printf("\t\t\t\t┌─────────────────────────────────────┐\n");  
	printf("\t\t\t\t│ 南校区医务室 │ 升华学生公寓19栋     │\n");
	printf("\t\t\t\t│─────────────────────────────────────│\n");
	printf("\t\t\t\t│ 新校区医务室 │ 教学楼B座一楼        │\n");
	printf("\t\t\t\t│─────────────────────────────────────│\n");
	printf("\t\t\t\t│    校本部    │ 校本部职工医院       │\n");
	printf("\t\t\t\t│─────────────────────────────────────│\n");
	printf("\t\t\t\t│   湘雅校区   │ 湘雅医院             │\n");
	printf("\t\t\t\t└─────────────────────────────────────┘\n");	
	system("pause");
	system("cls");
}

void traffic()								//交通
{
	printf("\t\t\t中南大学公交线路：中南大学有多条公交线路，包括南北、东西、环线、城际等\n");
	printf("\t\t\t中南大学校内交通：中南大学有美团以及哈啰共享单车，同学们也可自行购买自行车或电动车\n");
	printf("\t\t\t中南大学充电桩：新校区：教学楼B、C座楼下以及物理学院、机电学院等学院楼下\n");
	printf("\t\t\t                南校区：西南2门入口处，28栋旁等\n");
	printf("\t\t\t中南大学游玩交通：通往后湖小区的红绿灯处有3个地铁口\n");
	system("pause");
	system("cls");
}

int user_service()							//生活服务菜单
{	
	int i;

	while(1){
	printf("\t\t\t\t┌───────────────────────────────────────────────┐\n");  
	printf("\t\t\t\t│       欢迎来到中南大学校园生活服务系统        │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                    1.餐饮                     │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                    2.医疗                     │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                    3.交通                     │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                    0.返回                     │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t└───────────────────────────────────────────────┘\n");	
	printf("\t\t\t\t请选择你想了解的内容（输入相应数字）:");
	scanf("%d", &i);
	system("cls");
	switch(i)
		{
			case 0:user();break;
			case 1:eat();break;
			case 2:health();break;
			case 3:traffic();break;
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
	
		
	return 0;
}

void study_one()							//2401课表
	{
		printf("\t\t\t\t┌────────────────────────────────────────────────────────────────┐\n");  
		printf("\t\t\t\t│   星期一   │   星期二   │   星期三   │   星期四   │   星期五   │\n");
		printf("\t\t\t\t│────────────────────────────────────────────────────────────────│\n");
		printf("\t\t\t\t│高级程序设计│  大学英语  │   新生课   │高级程序设计│  大学英语  │\n");
		printf("\t\t\t\t│   B座219   │ 外语楼104  │   B座519   │   B座117   │ 外语楼135  │\n");
		printf("\t\t\t\t│────────────────────────────────────────────────────────────────│\n");
		printf("\t\t\t\t│            │  大学体育  │            │思想道德与法│  军事理论  │\n");
		printf("\t\t\t\t│            │ 校本部操场 │            │治  B座420  │   A座108   │\n");
		printf("\t\t\t\t│────────────────────────────────────────────────────────────────│\n");
		printf("\t\t\t\t│            │            │            │  高等数学  │            │\n");
		printf("\t\t\t\t│            │            │            │   C座104   │            │\n");
		printf("\t\t\t\t│────────────────────────────────────────────────────────────────│\n");
		printf("\t\t\t\t│  高等数学  │            │            │            │  高等数学  │\n");
		printf("\t\t\t\t│   B座413   │            │            │            │   C座104   │\n");
		printf("\t\t\t\t└────────────────────────────────────────────────────────────────┘\n");	
		system("pause");
		system("cls");
	}
	
void study_two()						//2402课表
	{
		printf("\t\t\t\t┌────────────────────────────────────────────────────────────────┐\n");  
		printf("\t\t\t\t│   星期一   │   星期二   │   星期三   │   星期四   │   星期五   │\n");
		printf("\t\t\t\t│────────────────────────────────────────────────────────────────│\n");
		printf("\t\t\t\t│高级程序设计│            │   新生课   │高级程序设计│  大学英语  │\n");
		printf("\t\t\t\t│   B座219   │            │   B座519   │   B座117   │ 外语楼446  │\n");
		printf("\t\t\t\t│────────────────────────────────────────────────────────────────│\n");
		printf("\t\t\t\t│            │            │            │思想道德与法│  军事理论  │\n");
		printf("\t\t\t\t│            │            │            │治  B座420  │   A座108   │\n");
		printf("\t\t\t\t│────────────────────────────────────────────────────────────────│\n");
		printf("\t\t\t\t│            │            │            │  高等数学  │            │\n");
		printf("\t\t\t\t│            │            │            │   C座104   │            │\n");
		printf("\t\t\t\t│────────────────────────────────────────────────────────────────│\n");
		printf("\t\t\t\t│  高等数学  │  大学体育  │            │  大学英语  │  高等数学  │\n");
		printf("\t\t\t\t│   B座413   │ 校本部操场 │            │ 外语楼118  │   C座104   │\n");
		printf("\t\t\t\t└────────────────────────────────────────────────────────────────┘\n");	
		system("pause");
		system("cls");
	}

int uclass()								//课表查询
{
	int i;
	printf("\t\t\t\t请输入你的班级（2401/2402):");
	scanf("%d",&i);
	system("cls");
	switch(i)
		{
			case 2401:study_one();break;
			case 2402:study_two();break;
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;		
		}
	return 0;
}

void mgrade()								//成绩录入
{
	char id[10];
	printf("\t\t\t\t请输入你的学号：");
	scanf("%s",id);
	getchar(); 
	for(int i=0;i<n;i++)
	{
		if(strcmp(id,stu[i].ID)==0)
		{
			printf("\t\t\t\t请按顺序输入高数、大英、编程成绩(用空格隔开)：");
			scanf("%f%f%f",&stu[i].grade[0],&stu[i].grade[1],&stu[i].grade[2]);
		
			FILE *p=fopen("学生信息.txt","ab+");
			if(p==NULL)
			exit(0); 
			fwrite(&stu[i],len1,1,p);
			fclose(p);
		}
		else
			printf("\t\t\t\t未找到该学生，无法录入成绩，请重新输入");
	}
	system("pause");
}

void ugrade()								//成绩查询
{
	char id[10];
	printf("\t\t\t\t请输入你的学号：");
	scanf("%s",id);
	getchar(); 
	if(n>0)
	{
	for(int i=0;i<n;i++)
	{
		if(strcmp(id,stu[i].ID)==0)
		{
			printf("\t%s %s\t高数：%f\t大英：%f\t编程：%f\t",stu[i].ID,stu[i].Name,stu[i].grade[0],stu[i].grade[1],stu[i].grade[2]);
			printf("总分：%f",stu[i].grade[0]+stu[i].grade[1]+stu[i].grade[2]);
			printf("\n\t\t\t\t若无成绩信息，请等待");
			printf("\n");
		}
		else
			printf("\t\t\t\t未找到该学生，无法查询成绩，请重新输入\n");
	}
	}
	else
		printf("\t\t\t\t暂无成绩信息，无法查询，请等待\n");
	system("pause");
	
}

int igrade()								//成绩统计
{
	int i;
	if(n>0)
	{
		for(i=0;i<n;i++)
		{
			printf("\t%s %s\t高数：%f\t大英：%f\t编程：%f\t",stu[i].ID,stu[i].Name,stu[i].grade[0],stu[i].grade[1],stu[i].grade[2]);
			printf("总分：%f",stu[i].grade[0]+stu[i].grade[1]+stu[i].grade[2]);
			printf("\n");
		}
		system("pause");
		return 1;	
	}
	else
	{
		printf("\t\t\t\t无学生信息");
		system("pause");
		return 0;	
	} 
	
}

int user_study()							//用户教学管理系统菜单
{
	int i;

	while(1){
	printf("\t\t\t\t┌────────────────────────────────────────────┐\n");  
	printf("\t\t\t\t│       欢迎来到中南大学教学管理系统         │\n");
	printf("\t\t\t\t│                                            │\n");
	printf("\t\t\t\t│                 1.课表查询                 │\n");
	printf("\t\t\t\t│                                            │\n");
	printf("\t\t\t\t│                 2.成绩查询                 │\n");
	printf("\t\t\t\t│                                            │\n");
	printf("\t\t\t\t│                 0.返回                     │\n");
	printf("\t\t\t\t│                                            │\n");
	printf("\t\t\t\t└────────────────────────────────────────────┘\n");	
	printf("\t\t\t\t请选择你需要的操作（输入相应数字）:");
	scanf("%d", &i);
	system("cls");
	switch(i)
		{
			case 0:user();break;
			case 1:uclass();break;
			case 2:ugrade();system("cls");break;
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
	return 0;
}

int manager_study()							//管理员教学管理系统菜单
{
	int i;

	while(1){
	printf("\t\t\t\t┌───────────────────────────────────────────┐\n");  
	printf("\t\t\t\t│       欢迎来到中南大学教学管理系统        │\n");
	printf("\t\t\t\t│                                           │\n");
	printf("\t\t\t\t│                1.课表查询                 │\n");
	printf("\t\t\t\t│                                           │\n");
	printf("\t\t\t\t│                2.成绩录入                 │\n");
	printf("\t\t\t\t│                                           │\n");
	printf("\t\t\t\t│                3.成绩统计                 │\n");
	printf("\t\t\t\t│                                           │\n");
	printf("\t\t\t\t│                4.成绩查询                 │\n");
	printf("\t\t\t\t│                                           │\n");
	printf("\t\t\t\t│                0.返回                     │\n");
	printf("\t\t\t\t│                                           │\n");
	printf("\t\t\t\t└───────────────────────────────────────────┘\n");	
	printf("\t\t\t\t请选择你需要的操作（输入相应数字）:");
	scanf("%d", &i);
	system("cls");
	switch(i)
		{
			case 0:user();break;
			case 1:uclass();break;
			case 2:mgrade();system("cls");break;
			case 3:igrade();system("cls");break;
			case 4:ugrade();system("cls");break;
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
	return 0;
}

int user_help()								//用户新生报道管理系统菜单
{
	int i;

	while(1){
	printf("\t\t\t\t┌───────────────────────────────────────────────┐\n");  
	printf("\t\t\t\t│       欢迎来到中南大学新生报道管理系统        │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│              1.新生报到信息录入               │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│              2.新生报到信息修改               │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│              0.返回                           │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t└───────────────────────────────────────────────┘\n");	
	printf("\t\t\t\t请选择你需要的操作（输入相应数字）:");
	scanf("%d", &i);
	system("cls");
	switch(i)
		{
			case 0:user();break;
			case 1:user_input();system("cls");break;
			case 2:user_revise();system("cls");break;
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
	return 0;
}

int manager_help()							//管理员新生报道管理系统菜单
{
	int i;

	while(1){
	printf("\t\t\t\t┌───────────────────────────────────────────────┐\n");  
	printf("\t\t\t\t│       欢迎来到中南大学新生报道管理系统        │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│              1.新生报到信息删除               │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│              2.新生报到信息查询               │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│              3.新生报到信息统计               │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│              4.新生报到信息清空               │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│              0.返回                           │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t└───────────────────────────────────────────────┘\n");	
	printf("\t\t\t\t请选择你需要的操作（输入相应数字）:");
	scanf("%d", &i);
	system("cls");
	switch(i)
		{
			case 0:manager();break;
			case 1:deletee();system("cls");break;
			case 2:find();system("cls");break;
			case 3:present();system("cls");break;
			case 4:enter();system("cls");break;
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
	return 0;
}

void user_library()								//用户图书管理系统菜单
{
	int i;
	while(1){
	printf("\t\t\t\t┌───────────────────────────────────────────────┐\n"); 
	printf("\t\t\t\t│         欢迎来到中南大学图书借阅系统          │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                  1.图书查询                   │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                  2.图书借阅                   │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                  3.图书归还                   │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                  4.图书推荐                   │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                  0.返回                       │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t└───────────────────────────────────────────────┘\n");
	printf("\t\t\t\t请选择你需要的操作（输入相应数字）:");
	scanf("%d", &i);
	system("cls");
	switch(i)
		{	case 0:user();break;
			case 1:lib_research();system("cls");break;
			case 2:lib_borrow();system("cls");break;
			case 3:lib_return();system("cls");break;
			case 4:lib_recommend();break;
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
}

void manager_library()						//管理员图书管理系统菜单
{
	int i;
	while(1)
	{
	printf("\t\t\t\t┌───────────────────────────────────────────────┐\n"); 
	printf("\t\t\t\t│         欢迎来到中南大学图书借阅系统          │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                  1.图书收入                   │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                  2.图书统计                   │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                  0.返回                       │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t└───────────────────────────────────────────────┘\n");
	printf("\t\t\t\t请选择你需要的操作（输入相应数字）:");
	scanf("%d", &i);
	system("cls");
	switch(i)
		{	case 0:manager();break;
			case 1:lib_add();system("cls");break;
			case 2:lib_present();system("cls");break;
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
}

int user()										//用户 菜单
{
	int i;
	while(1){
	printf("\t\t\t\t┌───────────────────────────────────────────────┐\n"); 
	printf("\t\t\t\t│       欢迎来到中南大学新生服务管理系统        │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                 1.新生报到                    │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                 2.校园生活服务                │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                 3.教学管理系统                │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                 4.图书借阅系统                │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                 0.返回                        │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t└───────────────────────────────────────────────┘\n");
	printf("\t\t\t\t请选择你需要的操作（输入相应数字）:");
	scanf("%d", &i);
	system("cls");
	switch(i)
		{	case 0:menu();break;
			case 1:user_help();break;
			case 2:user_service();break;
			case 3:user_study();break;
			case 4:user_library();break;
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
	
	return 0;
}

int manager()							//管理员 菜单
{
	int i;
	while(1){
	printf("\t\t\t\t┌───────────────────────────────────────────────┐\n"); 
	printf("\t\t\t\t│       欢迎来到中南大学新生服务管理系统        │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│             1.新生信息查询与统计              │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│             2.教学管理系统                    │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│             3.图书借阅管理系统                │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│             0.返回                            │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t└───────────────────────────────────────────────┘\n");
	printf("\t\t\t\t请选择你需要的操作（输入相应数字）:");
	scanf("%d", &i);
	system("cls");
	switch(i)
		{	case 0:menu();break;
			case 1:manager_help();break;
			case 2:manager_study();break;
			case 3:manager_library();break;
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
	return 0;
}

int menu()									//登录口 
{
	while(1){
	
	system("cls");
	printf("\t\t\t\t┌───────────────────────────────────────────────┐\n");     
	printf("\t\t\t\t│       欢迎来到中南大学新生服务管理系统        │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                    你是？                     │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                   1.用户                      │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                   2.管理员                    │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t│                   0.退出系统                  │\n");
	printf("\t\t\t\t│                                               │\n");
	printf("\t\t\t\t└───────────────────────────────────────────────┘\n");
	printf("\t\t\t\t请选择你需要的操作（输入相应数字）:");
	int i=-1;
	scanf("%d", &i);
	system("cls");
	switch(i)
		{	
			case 0:system("cls");system("pause");break;
			case 1:user();break;
			case 2:manager();break; 
			default:printf("\n\t\t\t\t\t输入有误，请重新输入\n");system("pause");system("cls");break;
		}
	}
	return 0;
}

void Load1()							//学生信息储存
{
	FILE *p=fopen("学生信息.txt","r");
	if(p==NULL)
	{
		exit(0);
	}
	fseek(p,0,SEEK_END);
	int size=(int)ftell(p)/len1;
	fseek(p,0,SEEK_SET);
	for(int i=0;i<size;i++)
	{
		fread(&stu[i],len1,1,p);
	}
	n=size;
	fclose(p);
}

void Load2()							//图书信息储存
{
	FILE *q=fopen("图书信息.txt","r");
	if(q==NULL)
	{
		exit(0);
	}
	fseek(q,0,SEEK_END);
	int size=(int)ftell(q)/len2;
	fseek(q,0,SEEK_SET);
	for(int i=0;i<size;i++)
	{
		fread(&book[i],len2,1,q);
	}
	n2=size;
	fclose(q);
}

int main() 
{	
	Load1();							//导入信息
	Load2();
	menu();								//显示菜单
	return 0;
}
