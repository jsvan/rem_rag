

class Chunk:
    def chunk(self, doc, head=""):
        paragraphs = self.split_paragraphs(doc, head)
        sentences = [p.split('. ') for p in paragraphs]
        return {'paragraphs': paragraphs,
                'sentences': sentences}



    """
    jsv: divide long paragraphs kinda recursively, but with stack
    The second great problem of universal significance with relation to which the United States has both a duty and a rare opportunity for constructive action is the restructuring of the international community and the development of the full potential of the United Nations. The effort to achieve a world made up exclusively of sovereign entities, all completely equal in status; the absolute quality of the modern concept of sovereignty; the increasing fragmentation of the international community; the consequent phenomenon of the mini-state-an entity saddled with a modality of participation in international life to the demands of which its resources are patently inadequate; the damage done to international parliamentarianisni by the wild incongruities produced by the principle of "one country, one vote;" the contradiction involved in this steady multiplication of sovereignties in certain parts of the world in an age when governments elsewhere-governments of greater age and more mature understanding-are trying precisely to bridge the rigidities of sovereignty and to recognize a higher and more unified pattern of obligations: all these factors call out for the sort of study of the problem, and leadership in attacking it, which the United States is outstandingly equipped to give. The failure to find reasonable answers to these questions has already had an adverse effect on the United Nations and has limited the contribution-so desperately needed-which that organization should be capable of making to the improvement of international life.
    """
    def split_paragraphs(self, document, head="", maxlen=250):
        """
        jsv: returns two substrings, cut on middle-ist period.
        """
        def split_long_paragraph(text):
            mid = len(text) // 2
            loc = mid
            for diff in range(mid):
                loc = mid + diff
                if text[loc] == '.':
                    break
                loc = mid - diff
                if text[loc] == '.':
                    break
            return text[:loc + 1], text[loc + 1:]

        good = []
        bad = document.split('\n')
        bad.reverse()

        while len(bad) > 0:
            test_subject = bad.pop().strip()
            if len(test_subject.split(' ')) > maxlen:
                s1, s2 = split_long_paragraph(test_subject)
                if len(s1) == 0 or len(s2) == 0:
                    s1 = s1.replace(';', '.')
                    s2 = s2.replace(';', '.')
                bad.append(s1)
                bad.append(s2)
            elif len(test_subject) < 150:
                pass
            else:
                good.append(head + test_subject)  # .encode('ascii',errors='ignore'))

        return good



"""

Maybe worth chunking on sentences and on paragraphs, embedding both. With both, connect them to a summarization of the document.
On RAG the doc summary will be concated with the similar chunks, to add context during generation. 


"""
