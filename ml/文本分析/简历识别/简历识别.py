#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import sys
import re

reload(sys)
sys.setdefaultencoding("utf-8")

cv_basic_patterns = {"name": "����", "phone": "�ֻ�|�绰", "email": "����|e-mail|mail|email", "age": "����|��",
                     "address": "ͨѶ��ַ|��ס��", "gender": "�Ա�|��|Ů"}
cv_edu_patterns = {"university": "��ҵԺУ|��������", "major": "רҵ", "degree": "ѧ��|��ר|ר��|˶ʿ|��ʿ|�о���"}
cv_job_patterns = {"evaluation": "��������|��������|�������|��Ȥ", "career": "��ְ����|ӦƸְλ|��ְ����|ְλ", "work": "��������|��������|����ְ��",
                   "project": "��Ŀ����"}

content = "��������С��  �������ޣ�10�����Ϲ�������  ���У��㶫 ������  �Ա�Ů  �������ڣ�1975-07-01  �ֻ���13825267167  ���䣺xiaodaizhou@163.com  ְλ������    ��������    ѧУ������ʡ����ʦ���ߵ�ר��ѧУ  ��ѧרҵ����������ѧ  ѧ�������ƣ���ҵ��ݣ�1998    ��������    ��˾������ΰ��(����)��̩�������޹�˾  ְλ����Ա  ��ʼ���ڣ�1996-03���������ڣ�1998-08  �������ݣ�  �� �칫�� �����ļ��浵���ɳ����������ֱ����쵼����ʹ��  ������������ϸ���ܡ�      ��˾������ɣ������ͨ�����޹�˾  ְλ���鳤  ��ʼ���ڣ�2000-03���������ڣ�2002-07  �������ݣ�  IQC a.����IQC���н��ϣ��ܽ��ϣ������ϣ���װ���ϵ��������ࣩ�ļ���  b.IQC����Ա��Ĺ���������Э����IQC�쳣�ĳ���ȷ���봦��IQC���ֱ��������  (2)     ��˾��˹������ʾ�Ƽ�(����)���޹�˾  ְλ������  ��ʼ���ڣ�2005-05���������ڣ�2007-11  �������ݣ�  �� ��� a.�ƻ��ı���������Э�����ƶ��������ƻ����ܽ��н��ڼƻ��ı���,������������,��������,Э  ����������/����/�豸����״���������ƻ�����Чִ��,ͻ���¼��쳣����Ĵ���  b.�����۵�λ��ͨ����FORECAST��Ϣ������ά����ϵͳ  c. ȱ�ϴ���:���������ƻ���������ĵ����ƻ��������������Ϲ�Ӧ˳��������ȱ�ϵĽ��ڻظ���  �����������ϵĿ��ٴ���ȱ��ԭ��������ܽἰ������ز��Ÿ���  d. Э����ϵ:��ǿ���������ɹ� ����Ȳ��ŵ���ϵ������������Ϣ����ʱ���ڿ��ˮƽ,�� ͬŬ����ɸ��������  e.������Э�ӹ��������Ͻ���  f.��������      ��˾����������άѶ���ӿƼ����޹�˾  ְλ������  ��ʼ���ڣ�2007-12���������ڣ�2009-01  �������ݣ�  �� PMC 1.�������幤������Ĵ�����빩Ӧ�̼����ż�Ĺ�ͨ��Э��  2.���빫˾��۹���Ͳ����ƶ��������������޶�������ҵ���̺͹����ƶ�  3.�ƶ������ڲ��̡��С����ڹ����ƻ���������ʵʩ�����������ָ���  4.��֯�����������ƻ������Ͽ��ơ����ϲɹ�����Ӧ�̿����ƻ��ʹ�ɽ���  5.���빩Ӧ�̵Ŀ��ˡ������������ز���ȷ���������,����������Ӧ���ش��쳣  6.�����ļ����ϼ����ݵ�ǩ��������  7.��������Ա�ļ�Ч���������ѵ����"

jd_include_keys = {"cv": "��λְ��|��ְҪ��|��ְ�ʸ�|����Ҫ��|����Ҫ��|ְ������|��λҪ��|��λ����|��λ����|ְλ����"}
cv_include_keys = {"jd": "��ְ����|��ְ״̬|��������|��������"}


def isMatch(pattern, text):
    if re.compile(pattern).search(text):
        return 1
    else:
        return 0


def cvMatchFlow(content):
    cv_basic_matches = {}
    cv_basic_matches["total"] = 0
    cv_edu_matches = {}
    cv_edu_matches["total"] = 0
    cv_job_matches = {}
    cv_job_matches["total"] = 0
    cv_key_matches = {}

    cv_key_matches["total"] = 0
    jd_key_matches = {}
    jd_key_matches["total"] = 0

    for k, v in cv_basic_patterns.items():
        cv_basic_matches[k] = isMatch(v, content)
        cv_basic_matches["total"] = cv_basic_matches[k] + cv_basic_matches["total"]

    for k, v in cv_edu_patterns.items():
        cv_edu_matches[k] = isMatch(v, content)
        cv_edu_matches["total"] = cv_edu_matches[k] + cv_edu_matches["total"]

    for k, v in cv_job_patterns.items():
        cv_job_matches[k] = isMatch(v, content)
        cv_job_matches["total"] = cv_job_matches[k] + cv_job_matches["total"]

    for k, v in cv_include_keys.items():
        cv_key_matches[k] = isMatch(v, content)
        cv_key_matches["total"] = cv_key_matches[k] + cv_key_matches["total"]

    for k, v in jd_include_keys.items():
        jd_key_matches[k] = isMatch(v, content)
        jd_key_matches["total"] = jd_key_matches[k] + jd_key_matches["total"]

        return cv_basic_matches, cv_edu_matches, cv_job_matches, cv_key_matches, jd_key_matches


cv_basic_matches, cv_edu_matches, cv_job_matches, cv_key_matches, jd_key_matches = cvMatchFlow(content)

print(cv_basic_matches)
print(cv_edu_matches)
print(cv_job_matches)
print(cv_key_matches)
print(jd_key_matches)
