#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <climits>
using namespace std;

const unordered_map<char, double> english_letter_frequencies = {
    {'a', 0.08167}, {'b', 0.01492}, {'c', 0.02782}, {'d', 0.04253},
    {'e', 0.12702}, {'f', 0.02228}, {'g', 0.02015}, {'h', 0.06094},
    {'i', 0.06966}, {'j', 0.00153}, {'k', 0.00772}, {'l', 0.04025},
    {'m', 0.02406}, {'n', 0.06749}, {'o', 0.07507}, {'p', 0.01929},
    {'q', 0.00095}, {'r', 0.05987}, {'s', 0.06327}, {'t', 0.09056},
    {'u', 0.02758}, {'v', 0.00978}, {'w', 0.02360}, {'x', 0.00150},
    {'y', 0.01974}, {'z', 0.00074}
};

unordered_map<char, int> letter_frequency(const string &text) {
    unordered_map<char, int> frequencies;
    for (char c : text) {
        if (isalpha(c)) {
            c = tolower(c);
            frequencies[c]++;
        }
    }
    return frequencies;
}

double score_plaintext(const string &text) {
    auto freq = letter_frequency(text);
    double total = 0;
    for (auto &p : freq) {
        total += p.second;
    }
    if (total == 0) return 0;
    double score = 0;
    double total_freq = 0;
    for (auto &pair : english_letter_frequencies) {
        total_freq += pair.second;
    }
    for (auto &pair : freq) {
        char c = pair.first;
        double observed = (double)pair.second / total;
        double expected = english_letter_frequencies.find(c) != english_letter_frequencies.end() ?
            english_letter_frequencies.at(c) : 0;
        score += (observed - expected) * (observed - expected) / expected;
    }
    return -score;
}

string brute_force_caesar(const string &ciphertext) {
    int max_score = INT_MIN;
    string best_plaintext;
    int shift = 0;
    for (int key = 0; key < 26; key++) {
        string plaintext;
        for (char c : ciphertext) {
            if (isalpha(c)) {
                char base = isupper(c) ? 'A' : 'a';
                char shifted = (c - base - key + 26) % 26 + base;
                plaintext += shifted;
            } else {
                plaintext += c;
            }
        }
        double current_score = score_plaintext(plaintext);
        if (current_score > max_score) {
            max_score = current_score;
            best_plaintext = plaintext;
            shift = key;
        }
    }
    return best_plaintext;
}

int main() {
    string ciphertext = "qnhiu tj oni b itpx nw qtwx; ti tj b jibix nw ptoy; ti tj oni b pbiixk nw knjd zuxxrj, kxy qtmj boy jhmmqx roxxj; ti tj b pbiixk nw iux ftqq, b lhbqtid nw iux tpbvtobitno, b gtvnk nw iux xpnitnoj; ti tj iux wkxjuoxjj nw iux yxxm jmktovj nw qtwx.qnhiu pxboj b ixpmxkbpxoibq mkxynptobozx nw znhkbvx ngxk itptytid, nw iux bmmxitix wnk bygxoihkx ngxk iux qngx nw xbjx. vutj nwixo xetjij to b pbo nw 60 pnkx iubo b and nw 20. bnanyd vknfj nqy pxkxqd ad b ohpaxk nw dxbkj. sx vknf nqy ad yxjxkitov nhk tyxbqj.qxbkj pbd fktorqx iux jrto, ahi in vtgx hm xoiuhjtbjp fktorqxj iux jnhq. snkkd, wxbk, jxqw-ytjikhji anfj iux uxbki boy ihkoj iux jmtkti abzr in yhji.suxiuxk 60 nk 16, iuxkx tj to xgxkd uhpbo axtov’j uxbki iux qhkx nw fnoyxkj, iux howbtqtov bmmxitix wnk fubi’j oxei boy iux snd nw iux vbpx nw qtgtov. go iux zxoixk nw dnhk uxbki boy pd uxbki, iuxkx tj b ftkxqxjj jibitno; jn qnov bj ti kxzxtgxj pxjjbvxj nw axbhid, unmx, znhkbvx boy mnfxk wknp pbo boy wknp iux towtotix, jn qnov bj dnh bkx dnhov.suxo dnhk bxktbqj bkx ynfo, boy dnhk jmtkti tj zngxkxy ftiu jonfj nw zdotztjp boy iux tzx nw mxjjtptjp, iuxo dnh’gx vknfo nqy, xgxo bi 20; ahi bj qnov bj dnhk bxktbqj bkx hm, in zbizu fbgxj nw nmitptjp, iuxkx’j unmx dnh pbd ytx dnhov bi 80.oqq nw hj ubgx kxby iuktqqtov jinktxj to futzu iux uxkn uby noqd b qtptixy boy jmxztwtxy itpx in qtgx. wnpxitpxj ti fbj bj qnov bj b dxbk, jnpxitpxj bj junki bj 24 unhkj. nhi bqfbdj fx fxkx toixkxjixy to ytjzngxktov shji unf iux ynnpxy uxkn zunjx in jmxoy utj qbji ybdj nk utj qbji unhkj. g jmxbr, nw znhkjx, nw wkxx pxo fun ubgx b zuntzx, oni znoyxpoxy zktptobqj funjx jmuxkx nw bzitgtitxj tj jiktziqd yxqtptixy.whzu jinktxj jxi hj iutortov, fnoyxktov fubi fx junhqy yn hoyxk jtptqbk ztkzhpjibozxj. subi xgxoij, fubi xemxktxozxj, fubi bjjnztbitnoj junhqy fx zknfy toin iunjx qbji unhkj bj pnkibq axtovj, fubi kxvkxij?wnpxitpxj g ubgx iunhvui ti fnhqy ax bo xezxqqxoi khqx in qtgx xbzu ybd bj tw fx junhqy ytx inpnkknf. whzu bo biitihyx fnhqy xpmubjtcx jubkmqd iux gbqhxj nw qtwx. sx junhqy qtgx xbzu ybd ftiu vxoiqxoxjj, gtvnk boy b rxxooxjj nw bmmkxztbitno futzu bkx nwixo qnji fuxo itpx jikxizuxj axwnkx hj to iux znojiboi mbonkbpb nw pnkx ybdj boy pnoiuj boy dxbkj in znpx. vuxkx bkx iunjx, nw znhkjx, fun fnhqy bynmi iux kmtzhkxbo pniin nw “kbi, yktor, boy ax pxkkd”. nhi pnji mxnmqx fnhqy ax zubjixoxy ad iux zxkibtoid nw tpmxoytov yxbiu.go jinktxj iux ynnpxy uxkn tj hjhbqqd jbgxy bi iux qbji ptohix ad jnpx jiknrx nw wnkihox, ahi bqpnji bqfbdj utj jxojx nw gbqhxj tj zubovxy. hx axznpxj pnkx bmmkxztbitgx nw iux pxbotov nw qtwx boy tij mxkpboxoi jmtktihbq gbqhxj. gi ubj nwixo axxo onixy iubi iunjx fun qtgx, nk ubgx qtgxy, to iux jubynf nw yxbiu aktov b pxqqnf jfxxioxjj in xgxkdiutov iuxd yn.cnji nw hj, unfxgxk, ibrx qtwx wnk vkboixy. sx ronf iubi nox ybd fx phji ytx, ahi hjhbqqd fx mtzihkx iubi ybd bj wbk to iux whihkx. suxo fx bkx to ahndboi uxbqiu, yxbiu tj bqq ahi hotpbvtobaqx. sx jxqynp iutor nw ti. vux ybdj jikxizu nhi to bo xoyqxjj gtjib. wn fx vn banhi nhk mxiid ibjrj, ubkyqd bfbkx nw nhk qtjiqxjj biitihyx infbky qtwx.vux jbpx qxiubkvd, g bp bwkbty, zubkbzixktcxj iux hjx nw bqq nhk wbzhqitxj boy jxojxj. aoqd iux yxbw bmmkxztbix uxbktov, noqd iux aqtoy kxbqtcx iux pbotwnqy aqxjjtovj iubi qtx to jtvui. zbkitzhqbkqd ynxj iutj najxkgbitno bmmqd in iunjx fun ubgx qnji jtvui boy uxbktov to byhqi qtwx. nhi iunjx fun ubgx oxgxk jhwwxkxy tpmbtkpxoi nw jtvui nk uxbktov jxqynp pbrx iux whqqxji hjx nw iuxjx aqxjjxy wbzhqitxj. vuxtk xdxj boy xbkj ibrx to bqq jtvuij boy jnhoyj ubctqd, ftiunhi znozxoikbitno boy ftiu qtiiqx bmmkxztbitno. gi tj iux jbpx nqy jinkd nw oni axtov vkbixwhq wnk fubi fx ubgx hoitq fx qnjx ti, nw oni axtov znojztnhj nw uxbqiu hoitq fx bkx tqq.g ubgx nwixo iunhvui ti fnhqy ax b aqxjjtov tw xbzu uhpbo axtov fxkx jiktzrxo aqtoy boy yxbw wnk b wxf ybdj bi jnpx itpx yhktov utj xbkqd byhqi qtwx. lbkroxjj fnhqy pbrx utp pnkx bmmkxztbitgx nw jtvui; jtqxozx fnhqy ixbzu utp iux sndj nw jnhoy."; // "hello world" shifted by 3
    string plaintext = brute_force_caesar(ciphertext);
    cout << "Plaintext: " << plaintext << endl;
    return 0;
}