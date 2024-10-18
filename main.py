class Person:
    def __call__(self, name):
        print("__call__"+"hello "+name)

    def hello(self,name):
        print("hello "+name)

person=Person()
person("zhangsan") #使用call函数调用
person.hello("lisi")