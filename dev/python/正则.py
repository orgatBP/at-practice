//match����ֻ���RE�ǲ�����string�Ŀ�ʼλ��ƥ�䣬 search��ɨ������string����ƥ��
def isMatch(pattern,text):
    if re.compile(pattern).search(text):
        return 1
    else:
        return 0