using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Baza.MailFetching.Services
{
    public class ResumeDetector
    {
        private static readonly Regex _WhiteBlankRegex = new Regex(@"\s+", RegexOptions.Compiled);
        private static readonly IDictionary<string, string[]> _Patterns = new Dictionary<string, string[]>
        {
            { "main", new []{ "��ְ����", "��ְ״̬", "��������", "��������", "��������", "ʵϰ����", "����нˮ", "��ס��", "��������" } },
            { "base", new []{ "����", "�ֻ�", "�绰", "����", "e-mail", "����", "����",
                "ͨѶ��ַ", "��ס��", "����", "�Ա�", "��", "Ů", "����", "�ʼ�" } },
            { "education", new []{ "��ҵѧУ", "��ҵԺУ", "��������", "רҵ", "ѧ��", "��ר", "ר��", "����", "˶ʿ", "��ʿ", "�о���" } },
            { "job", new []{ "��ְ����", "ӦƸְλ", "��ְ����", "��������", "��������", "ְλ",
                "��Ŀ����", "��Ŀ", "����ְ��", "�����ص�", "��������", "��������", "���˼���",
                "���˾���", "ʵϰ����", "��������", "�������", "��Ȥ", "����ҵ��", "��������" } }
        };

        public static bool Detect(string text)
        {
            if (!string.IsNullOrEmpty(text))
            {
                text = _WhiteBlankRegex.Replace(text, "");
                if (string.IsNullOrEmpty(text))
                    return false;

                text = text.ToLower();
                var baseItems = _Patterns["base"].Select(i => text.IndexOf(i)).Where(i => i >= 0).ToArray();
                var educationItems = _Patterns["education"].Select(i => text.IndexOf(i)).Where(i => i >= 0).ToArray();
                var jobItems = _Patterns["job"].Select(i => text.IndexOf(i)).Where(i => i >= 0).ToArray();
                var half = text.Length / 2;
                if (baseItems.Length > 0 && jobItems.Length > 0)
                {
                    if (baseItems.Min() <= jobItems.Min() && baseItems.Min() < half)
                        return true;

                    if (educationItems.Length > 0 && educationItems.Min() < half && baseItems.Min() < educationItems.Min())
                        return true;

                    return _Patterns["main"].Any(text.Contains);
                }

                if (jobItems.Length > 0 && educationItems.Length > 0 && _Patterns["main"].Any(text.Contains))
                    return true;

                if (baseItems.Length >= 2 && jobItems.Length == 0 && educationItems.Length >= 2)
                    return true;

                if (jobItems.Length >= 2 && educationItems.Length >= 2)
                    return true;
            }
            return false;
        }
    }
}